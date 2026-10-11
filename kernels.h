// kernels.h - transformer building blocks: softmax, LayerNorm and multi-head
// self-attention, each as a readable scalar reference plus a NEON version.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "deps/gemm.h"
#include "deps/util.h"

namespace tk {

// ===========================================================================
// exp(x) for four floats.
// Range reduction x = n*ln2 + r with |r| <= ln2/2, a degree-5 polynomial for
// exp(r) (Cephes coefficients), then scaling by 2^n through the exponent bits.
// Maximum relative error is about 2e-7 (roughly 2 ulp).
// ===========================================================================
#if defined(__ARM_NEON)
static inline float32x4_t exp_ps(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));  // keep 2^n a normal float
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    // ln2 split in two parts so that n*ln2 is subtracted without rounding error
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    const float32x4_t e = vaddq_f32(vfmaq_f32(r, p, vmulq_f32(r, r)), vdupq_n_f32(1.0f));  // 1 + r + r^2 * p(r)
    const int32x4_t pow2n = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(e, vreinterpretq_f32_s32(pow2n));
}
#endif

// ===========================================================================
// Softmax over n values.
// ===========================================================================

// Textbook formula. exp() overflows to inf once x > 88.7, and inf/inf = NaN.
inline void softmax_naive(const float* x, float* y, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += (y[i] = std::exp(x[i]));
    for (int i = 0; i < n; ++i) y[i] /= sum;
}

// Stable: softmax is invariant to shifting every input, so subtract the
// maximum. The largest exponent is then exactly 0: nothing overflows and the
// sum is at least 1, so the division is always safe.
inline void softmax_stable(const float* x, float* y, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += (y[i] = std::exp(x[i] - m));
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) y[i] *= inv;
}

// The stable algorithm with all three passes vectorised. `x` and `y` may alias.
inline void softmax_simd(const float* x, float* y, int n) {
#if defined(__ARM_NEON)
    int i = 0;
    float m = x[0];
    if (n >= 16) {
        float32x4_t m0 = vld1q_f32(x), m1 = vld1q_f32(x + 4), m2 = vld1q_f32(x + 8), m3 = vld1q_f32(x + 12);
        for (i = 16; i + 16 <= n; i += 16) {
            m0 = vmaxq_f32(m0, vld1q_f32(x + i));
            m1 = vmaxq_f32(m1, vld1q_f32(x + i + 4));
            m2 = vmaxq_f32(m2, vld1q_f32(x + i + 8));
            m3 = vmaxq_f32(m3, vld1q_f32(x + i + 12));
        }
        m = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
    }
    for (; i < n; ++i) m = std::max(m, x[i]);

    const float32x4_t mv = vdupq_n_f32(m);
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0;
    for (i = 0; i + 8 <= n; i += 8) {
        const float32x4_t e0 = exp_ps(vsubq_f32(vld1q_f32(x + i), mv));
        const float32x4_t e1 = exp_ps(vsubq_f32(vld1q_f32(x + i + 4), mv));
        vst1q_f32(y + i, e0);
        vst1q_f32(y + i + 4, e1);
        s0 = vaddq_f32(s0, e0);
        s1 = vaddq_f32(s1, e1);
    }
    float sum = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; ++i) sum += (y[i] = std::exp(x[i] - m));

    const float inv = 1.0f / sum;
    for (i = 0; i + 4 <= n; i += 4) vst1q_f32(y + i, vmulq_n_f32(vld1q_f32(y + i), inv));
    for (; i < n; ++i) y[i] *= inv;
#else
    softmax_stable(x, y, n);
#endif
}

// ===========================================================================
// LayerNorm over the last dimension: y = (x - mean) / sqrt(var + eps) * gamma + beta
// ===========================================================================

// One-pass variance, var = E[x^2] - E[x]^2. Cheap, but it subtracts two
// nearly equal numbers when |mean| >> std: catastrophic cancellation.
inline void layernorm_onepass(const float* x, const float* gamma, const float* beta, float* y, int n,
                              float eps = 1e-5f) {
    float s = 0.0f, ss = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i], ss += x[i] * x[i];
    const float mean = s / n, var = ss / n - mean * mean;
    const float inv = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}

// Two-pass: mean first, then the sum of squared deviations. Every term is
// non-negative, so nothing cancels.
inline void layernorm_stable(const float* x, const float* gamma, const float* beta, float* y, int n,
                             float eps = 1e-5f) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i];
    const float mean = s / n;
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += (x[i] - mean) * (x[i] - mean);
    const float inv = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}

// The two-pass algorithm with NEON. Four independent accumulators per
// reduction break the add dependency chain (and, as a bonus, sum pairwise,
// which is more accurate than one long serial sum).
inline void layernorm_simd(const float* x, const float* gamma, const float* beta, float* y, int n,
                           float eps = 1e-5f) {
#if defined(__ARM_NEON)
    int i = 0;
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= n; i += 16) {
        a0 = vaddq_f32(a0, vld1q_f32(x + i));
        a1 = vaddq_f32(a1, vld1q_f32(x + i + 4));
        a2 = vaddq_f32(a2, vld1q_f32(x + i + 8));
        a3 = vaddq_f32(a3, vld1q_f32(x + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; ++i) s += x[i];
    const float mean = s / n;
    const float32x4_t mv = vdupq_n_f32(mean);

    a0 = a1 = a2 = a3 = vdupq_n_f32(0.0f);
    for (i = 0; i + 16 <= n; i += 16) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(x + i), mv), d1 = vsubq_f32(vld1q_f32(x + i + 4), mv);
        const float32x4_t d2 = vsubq_f32(vld1q_f32(x + i + 8), mv), d3 = vsubq_f32(vld1q_f32(x + i + 12), mv);
        a0 = vfmaq_f32(a0, d0, d0);
        a1 = vfmaq_f32(a1, d1, d1);
        a2 = vfmaq_f32(a2, d2, d2);
        a3 = vfmaq_f32(a3, d3, d3);
    }
    float ss = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; ++i) ss += (x[i] - mean) * (x[i] - mean);
    const float inv = 1.0f / std::sqrt(ss / n + eps);

    const float32x4_t iv = vdupq_n_f32(inv);
    for (i = 0; i + 4 <= n; i += 4) {
        const float32x4_t t = vmulq_f32(vsubq_f32(vld1q_f32(x + i), mv), iv);
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(beta + i), t, vld1q_f32(gamma + i)));
    }
    for (; i < n; ++i) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
#else
    layernorm_stable(x, gamma, beta, y, n, eps);
#endif
}

// ===========================================================================
// Multi-head self-attention block (pre-LayerNorm, as in GPT-2 / ViT):
//
//   h = LayerNorm(x)
//   Q = h Wq, K = h Wk, V = h Wv                      (L x D each)
//   per head:  P = softmax(Q_h K_h^T / sqrt(d)),  O_h = P V_h
//   y = x + concat_h(O_h) Wo
//
// x is L x D (sequence length x model width), row-major. Weight matrices are
// D x D and applied on the right. Biases are omitted to keep the focus on the
// kernels.
// ===========================================================================
struct AttentionWeights {
    int D = 0, heads = 0;
    std::vector<float> Wq, Wk, Wv, Wo, gamma, beta;
};

struct AttentionScratch {
    util::avec<float> h, Q, K, V, O, proj, Kt, scores;
    size_t score_floats = 0;  // size of the score buffer actually used (per thread)
};

// Reference: plain loops, full L x L score matrix, scalar softmax.
inline void attention_naive(const AttentionWeights& w, int L, const float* x, float* y, AttentionScratch& s) {
    const int D = w.D, H = w.heads, d = D / H;
    const float scale = 1.0f / std::sqrt((float)d);
    s.h.resize((size_t)L * D), s.Q.resize((size_t)L * D), s.K.resize((size_t)L * D), s.V.resize((size_t)L * D);
    s.O.resize((size_t)L * D), s.proj.resize((size_t)L * D), s.scores.resize((size_t)L * L);
    s.score_floats = (size_t)L * L;
    for (int i = 0; i < L; ++i)
        layernorm_stable(x + (size_t)i * D, w.gamma.data(), w.beta.data(), s.h.data() + (size_t)i * D, D);
    gemm::naive(L, D, D, s.h.data(), D, w.Wq.data(), D, s.Q.data(), D);
    gemm::naive(L, D, D, s.h.data(), D, w.Wk.data(), D, s.K.data(), D);
    gemm::naive(L, D, D, s.h.data(), D, w.Wv.data(), D, s.V.data(), D);
    for (int hd = 0; hd < H; ++hd) {
        const int off = hd * d;
        for (int i = 0; i < L; ++i) {
            float* row = s.scores.data() + (size_t)i * L;
            for (int j = 0; j < L; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < d; ++k) acc += s.Q[(size_t)i * D + off + k] * s.K[(size_t)j * D + off + k];
                row[j] = acc * scale;
            }
            softmax_stable(row, row, L);
            for (int k = 0; k < d; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < L; ++j) acc += row[j] * s.V[(size_t)j * D + off + k];
                s.O[(size_t)i * D + off + k] = acc;
            }
        }
    }
    gemm::naive(L, D, D, s.O.data(), D, w.Wo.data(), D, s.proj.data(), D);
    for (size_t i = 0; i < (size_t)L * D; ++i) y[i] = x[i] + s.proj[i];
}

// Optimised: every matrix product goes through the packed GEMM, softmax and
// LayerNorm through the NEON kernels.
//
// `tile` bounds memory: instead of materialising the L x L score matrix, the
// queries are processed `tile` rows at a time. Each tile computes its
// (tile x L) scores, turns them into probabilities in place and immediately
// reduces them to (tile x d) outputs, so the scratch is O(tile * L) rather
// than O(L^2) and stays in cache while it is hot. tile >= L reproduces the
// untiled algorithm.
inline void attention_fast(const AttentionWeights& w, int L, const float* x, float* y, AttentionScratch& s,
                           int tile, bool parallel) {
    const int D = w.D, H = w.heads, d = D / H;
    const float scale = 1.0f / std::sqrt((float)d);
    tile = std::max(1, std::min(tile, L));
    s.h.resize((size_t)L * D), s.Q.resize((size_t)L * D), s.K.resize((size_t)L * D), s.V.resize((size_t)L * D);
    s.O.resize((size_t)L * D), s.proj.resize((size_t)L * D), s.Kt.resize((size_t)H * d * L);
    s.score_floats = (size_t)tile * L;

#pragma omp parallel for if (parallel) schedule(static)
    for (int i = 0; i < L; ++i)
        layernorm_simd(x + (size_t)i * D, w.gamma.data(), w.beta.data(), s.h.data() + (size_t)i * D, D);
    gemm::sgemm(L, D, D, s.h.data(), D, w.Wq.data(), D, s.Q.data(), D, parallel);
    gemm::sgemm(L, D, D, s.h.data(), D, w.Wk.data(), D, s.K.data(), D, parallel);
    gemm::sgemm(L, D, D, s.h.data(), D, w.Wv.data(), D, s.V.data(), D, parallel);

    // K_h^T (d x L) per head, with the 1/sqrt(d) scaling folded in so the
    // score matrix never needs a separate scaling pass.
    for (int hd = 0; hd < H; ++hd)
        for (int k = 0; k < d; ++k) {
            float* dst = s.Kt.data() + ((size_t)hd * d + k) * L;
            for (int j = 0; j < L; ++j) dst[j] = s.K[(size_t)j * D + hd * d + k] * scale;
        }

    // (head, query tile) pairs are independent: each thread owns one score tile.
    const int ntiles = (L + tile - 1) / tile;
#pragma omp parallel if (parallel)
    {
        util::avec<float> scores((size_t)tile * L);
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (int hd = 0; hd < H; ++hd)
            for (int t = 0; t < ntiles; ++t) {
                const int i0 = t * tile, rows = std::min(tile, L - i0), off = hd * d;
                // scores (rows x L) = Q_h[i0:i0+rows] (rows x d) * K_h^T (d x L)
                gemm::sgemm(rows, L, d, s.Q.data() + (size_t)i0 * D + off, D, s.Kt.data() + (size_t)hd * d * L, L,
                            scores.data(), L);
                for (int r = 0; r < rows; ++r) softmax_simd(scores.data() + (size_t)r * L, scores.data() + (size_t)r * L, L);
                // O_h[i0:i0+rows] (rows x d) = P (rows x L) * V_h (L x d)
                gemm::sgemm(rows, d, L, scores.data(), L, s.V.data() + off, D, s.O.data() + (size_t)i0 * D + off, D);
            }
    }

    gemm::sgemm(L, D, D, s.O.data(), D, w.Wo.data(), D, s.proj.data(), D, parallel);
    for (size_t i = 0; i < (size_t)L * D; ++i) y[i] = x[i] + s.proj[i];
}

}  // namespace tk

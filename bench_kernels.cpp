// bench_kernels.cpp - correctness, numerical-stability demonstrations and
// timings for softmax, LayerNorm and the attention block.
//
// usage: bench_kernels [results_dir] [attention_tile_rows]
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "kernels.h"

// Double-precision references.
static void softmax_ref(const float* x, double* y, int n) {
    double m = x[0], sum = 0.0;
    for (int i = 1; i < n; ++i) m = std::max(m, (double)x[i]);
    for (int i = 0; i < n; ++i) sum += (y[i] = std::exp((double)x[i] - m));
    for (int i = 0; i < n; ++i) y[i] /= sum;
}

static void layernorm_ref(const float* x, double* y, int n, double eps = 1e-5) {
    double s = 0.0, ss = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    const double mean = s / n;
    for (int i = 0; i < n; ++i) ss += (x[i] - mean) * (x[i] - mean);
    const double inv = 1.0 / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv;
}

template <class V>
static double max_err(const float* a, const V& ref, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = std::fabs((double)a[i] - ref[i]);
        m = std::isnan(d) ? INFINITY : std::max(m, d);
    }
    return m;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "";
    auto open = [&](const char* name) { return dir.empty() ? nullptr : std::fopen((dir + "/" + name).c_str(), "w"); };
    util::Rng rng(11);

    // ------------------------------------------------------------------ exp
    {
        double worst = 0.0;
        for (int i = 0; i < 2000000; ++i) {
            float x[4], e[4];
            for (float& v : x) v = rng.uniform(-87.0f, 88.0f);
            vst1q_f32(e, tk::exp_ps(vld1q_f32(x)));
            for (int k = 0; k < 4; ++k) worst = std::max(worst, std::fabs(e[k] / std::exp((double)x[k]) - 1.0));
        }
        std::printf("NEON exp: max relative error over 8M random inputs in [-87, 88] = %.2e\n\n", worst);
        if (FILE* f = open("exp.csv")) {
            std::fprintf(f, "max_rel_err\n%.3e\n", worst);
            std::fclose(f);
        }
    }

    // -------------------------------------------------------------- softmax
    {
        FILE* csv = open("softmax.csv");
        if (csv) std::fprintf(csv, "n,naive_ns,stable_ns,simd_ns,speedup,simd_max_err\n");
        std::printf("Softmax (ns per element)\n%8s %10s %10s %10s %9s %14s\n", "n", "naive", "stable", "NEON", "speedup",
                    "NEON max err");
        for (int n : {64, 256, 1024, 4096, 16384}) {
            const int rows = std::max(1, (1 << 18) / n);
            util::avec<float> x((size_t)rows * n), y((size_t)rows * n);
            util::fill_uniform(x.data(), x.size(), rng, -8.0f, 8.0f);
            std::vector<double> ref(n);
            double err = 0.0;
            auto run = [&](void (*f)(const float*, float*, int)) {
                return util::time_best([&] {
                    for (int r = 0; r < rows; ++r) f(x.data() + (size_t)r * n, y.data() + (size_t)r * n, n);
                }) / ((double)rows * n) * 1e9;
            };
            const double t_naive = run(tk::softmax_naive), t_stable = run(tk::softmax_stable), t_simd = run(tk::softmax_simd);
            for (int r = 0; r < rows; ++r) {
                softmax_ref(x.data() + (size_t)r * n, ref.data(), n);
                err = std::max(err, max_err(y.data() + (size_t)r * n, ref, n));
            }
            std::printf("%8d %10.3f %10.3f %10.3f %8.1fx %14.2e\n", n, t_naive, t_stable, t_simd, t_stable / t_simd, err);
            if (csv) std::fprintf(csv, "%d,%.4f,%.4f,%.4f,%.2f,%.3e\n", n, t_naive, t_stable, t_simd, t_stable / t_simd, err);
        }
        if (csv) std::fclose(csv);

        // Stability: logits of the size produced by an unnormalised network.
        const float big[6] = {1000.0f, 999.0f, 995.0f, 0.0f, -1000.0f, 90.0f};
        float yn[6], ys[6], yv[6];
        double ref[6];
        tk::softmax_naive(big, yn, 6);
        tk::softmax_stable(big, ys, 6);
        tk::softmax_simd(big, yv, 6);
        softmax_ref(big, ref, 6);
        std::printf("\nStability, x = [1000, 999, 995, 0, -1000, 90]\n  exact : %.6f %.6f %.6f %.1e %.1e %.1e\n", ref[0],
                    ref[1], ref[2], ref[3], ref[4], ref[5]);
        std::printf("  naive : %.6f %.6f %.6f ...\n  stable: %.6f %.6f %.6f ...\n  NEON  : %.6f %.6f %.6f ...\n\n", yn[0],
                    yn[1], yn[2], ys[0], ys[1], ys[2], yv[0], yv[1], yv[2]);
        if (FILE* f = open("softmax_stability.csv")) {
            std::fprintf(f, "method,y0,y1,y2\nexact,%.6f,%.6f,%.6f\nnaive,%f,%f,%f\nstable,%.6f,%.6f,%.6f\nneon,%.6f,%.6f,%.6f\n",
                         ref[0], ref[1], ref[2], yn[0], yn[1], yn[2], ys[0], ys[1], ys[2], yv[0], yv[1], yv[2]);
            std::fclose(f);
        }
    }

    // ------------------------------------------------------------ LayerNorm
    {
        FILE* csv = open("layernorm.csv");
        if (csv) std::fprintf(csv, "n,onepass_ns,stable_ns,simd_ns,speedup,simd_max_err\n");
        std::printf("LayerNorm (ns per element)\n%8s %10s %10s %10s %9s %14s\n", "n", "one-pass", "two-pass", "NEON",
                    "speedup", "NEON max err");
        for (int n : {64, 256, 768, 1024, 4096}) {
            const int rows = std::max(1, (1 << 18) / n);
            util::avec<float> x((size_t)rows * n), y((size_t)rows * n), gamma(n, 1.0f), beta(n, 0.0f);
            util::fill_uniform(x.data(), x.size(), rng, -3.0f, 3.0f);
            auto run = [&](void (*f)(const float*, const float*, const float*, float*, int, float)) {
                return util::time_best([&] {
                    for (int r = 0; r < rows; ++r)
                        f(x.data() + (size_t)r * n, gamma.data(), beta.data(), y.data() + (size_t)r * n, n, 1e-5f);
                }) / ((double)rows * n) * 1e9;
            };
            const double t1 = run(tk::layernorm_onepass), t2 = run(tk::layernorm_stable), t3 = run(tk::layernorm_simd);
            std::vector<double> ref(n);
            double err = 0.0;
            for (int r = 0; r < rows; ++r) {
                layernorm_ref(x.data() + (size_t)r * n, ref.data(), n);
                err = std::max(err, max_err(y.data() + (size_t)r * n, ref, n));
            }
            std::printf("%8d %10.3f %10.3f %10.3f %8.1fx %14.2e\n", n, t1, t2, t3, t2 / t3, err);
            if (csv) std::fprintf(csv, "%d,%.4f,%.4f,%.4f,%.2f,%.3e\n", n, t1, t2, t3, t2 / t3, err);
        }
        if (csv) std::fclose(csv);

        // Stability: small variance riding on a large offset.
        FILE* scsv = open("layernorm_stability.csv");
        if (scsv) std::fprintf(scsv, "offset,onepass_err,stable_err,simd_err\n");
        std::printf("\nStability, x = offset + noise in [-1, 1], n = 768 (max abs error vs float64)\n%10s %12s %12s %12s\n",
                    "offset", "one-pass", "two-pass", "NEON");
        const int n = 768;
        util::avec<float> noise(n), x(n), y(n), gamma(n, 1.0f), beta(n, 0.0f);
        util::fill_uniform(noise.data(), n, rng);
        std::vector<double> ref(n);
        for (float offset : {0.0f, 10.0f, 100.0f, 1000.0f, 10000.0f}) {
            for (int i = 0; i < n; ++i) x[i] = offset + noise[i];
            layernorm_ref(x.data(), ref.data(), n);
            tk::layernorm_onepass(x.data(), gamma.data(), beta.data(), y.data(), n);
            const double e1 = max_err(y.data(), ref, n);
            tk::layernorm_stable(x.data(), gamma.data(), beta.data(), y.data(), n);
            const double e2 = max_err(y.data(), ref, n);
            tk::layernorm_simd(x.data(), gamma.data(), beta.data(), y.data(), n);
            const double e3 = max_err(y.data(), ref, n);
            std::printf("%10.0f %12.2e %12.2e %12.2e\n", offset, e1, e2, e3);
            if (scsv) std::fprintf(scsv, "%.0f,%.3e,%.3e,%.3e\n", offset, e1, e2, e3);
        }
        if (scsv) std::fclose(scsv);
    }

    // ------------------------------------------------------------ attention
    {
        const int D = 256, H = 4, TILE = argc > 2 ? std::atoi(argv[2]) : 256;
        tk::AttentionWeights w;
        w.D = D, w.heads = H;
        for (auto* m : {&w.Wq, &w.Wk, &w.Wv, &w.Wo}) {
            m->resize((size_t)D * D);
            util::fill_uniform(m->data(), m->size(), rng, -0.08f, 0.08f);
        }
        w.gamma.assign(D, 1.0f);
        w.beta.assign(D, 0.0f);

        FILE* csv = open("attention.csv");
        if (csv) std::fprintf(csv, "L,naive_ms,full_ms,tiled_ms,tiled_omp_ms,full_scratch_mb,tiled_scratch_mb,max_err\n");
        std::printf("\nSelf-attention block, D = %d, %d heads, tile = %d query rows (ms)\n", D, H, TILE);
        std::printf("%6s %10s %12s %10s %12s %14s %14s %10s\n", "L", "naive", "GEMM+NEON", "tiled", "tiled+OMP",
                    "scores full", "scores tiled", "max err");
        for (int L : {128, 256, 512, 1024, 2048, 4096}) {
            util::avec<float> x((size_t)L * D), y_ref((size_t)L * D), y((size_t)L * D);
            util::fill_uniform(x.data(), x.size(), rng);
            tk::AttentionScratch s;
            const bool do_naive = L <= 1024;  // the reference is O(L^2 d) scalar work: skip the largest sizes
            double t_naive = -1.0;
            if (do_naive) t_naive = util::time_best([&] { tk::attention_naive(w, L, x.data(), y_ref.data(), s); }, 0.2, 1);
            omp_set_num_threads(1);
            const double t_full = util::time_best([&] { tk::attention_fast(w, L, x.data(), y.data(), s, L, false); });
            const double full_mb = s.score_floats * 4.0 / (1 << 20);
            if (!do_naive) y_ref = y;  // beyond the naive range, check tiled against untiled
            double err = do_naive ? util::max_abs_diff(y.data(), y_ref.data(), y.size()) : 0.0;
            const double t_tiled = util::time_best([&] { tk::attention_fast(w, L, x.data(), y.data(), s, TILE, false); });
            const double tiled_mb = s.score_floats * 4.0 / (1 << 20);
            err = std::max(err, util::max_abs_diff(y.data(), y_ref.data(), y.size()));
            omp_set_num_threads(omp_get_num_procs());
            const double t_omp = util::time_best([&] { tk::attention_fast(w, L, x.data(), y.data(), s, TILE, true); });
            err = std::max(err, util::max_abs_diff(y.data(), y_ref.data(), y.size()));

            if (do_naive)
                std::printf("%6d %10.2f", L, t_naive * 1e3);
            else
                std::printf("%6d %10s", L, "-");
            std::printf(" %12.2f %10.2f %12.2f %11.2f MB %11.2f MB %10.1e\n", t_full * 1e3, t_tiled * 1e3, t_omp * 1e3,
                        full_mb, tiled_mb, err);
            if (csv)
                std::fprintf(csv, "%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3e\n", L, do_naive ? t_naive * 1e3 : -1.0, t_full * 1e3,
                             t_tiled * 1e3, t_omp * 1e3, full_mb, tiled_mb, err);
        }
        if (csv) std::fclose(csv);
    }
    return 0;
}

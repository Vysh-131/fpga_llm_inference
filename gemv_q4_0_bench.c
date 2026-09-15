/*
 * gemv_q4_0_bench.c
 *
 * Standalone Q4_0 GEMV kernel benchmark — Phase 3 (isolated Pi kernel benchmark)
 *
 * Loads a raw Q4_0-quantized weight matrix in native ggml/GGUF block format,
 * a real fp32 activation vector, and a reference fp32 output vector (all
 * extracted in Phase 2 from a live Qwen2.5-3B forward pass). Computes
 * y = W @ x via on-the-fly Q4_0 dequantization, checks the result against
 * the reference output, and times N repeated runs to report median latency
 * and derived GOPS.
 *
 * This is deliberately standalone (no llama.cpp linkage) so the exact same
 * (W, x, y) problem can later be run on a GPU and on the PYNQ PL kernel for
 * an apples-to-apples comparison.
 *
 * Build (on the Pi):
 *   gcc -O3 -march=native -funroll-loops -o gemv_q4_0_bench gemv_q4_0_bench.c -lm
 *
 * Usage:
 *   ./gemv_q4_0_bench W.bin x.bin y_ref.bin [n_rows] [n_cols] [iterations]
 *
 * Defaults match the layer-15 ffn_down kernel used throughout this project:
 *   n_rows (output dim)     = 2048
 *   n_cols (input/contract) = 11008
 *   iterations              = 200
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>

#define QK4_0 32
#define BLOCK_BYTES 18   /* 2 bytes fp16 scale + 16 bytes packed 4-bit quants */

/* ---------------------------------------------------------------------
 * Portable IEEE-754 half -> single precision conversion.
 * Matches ggml_fp16_t exactly; written from scratch rather than relying
 * on a compiler-specific __fp16 type, so this file builds identically
 * on any target (Pi, x86 dev machine, etc.) for cross-checking.
 * --------------------------------------------------------------------- */
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign; /* +-0 */
        } else {
            /* subnormal half -> normalize into a normal float */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13); /* inf / nan */
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

typedef struct {
    uint16_t d;            /* fp16 block scale */
    uint8_t  qs[QK4_0/2];  /* 16 bytes = 32 packed 4-bit quants, 2 per byte */
} block_q4_0;

/*
 * Dequantize one 18-byte Q4_0 block into 32 floats.
 * ggml's packing is NOT sequential-pair (j, j+1) per byte — it's
 * (j, j+16): byte j's low nibble is element j, high nibble is element
 * j+16. Getting this backwards produces a plausible-looking but wrong
 * result, so it's called out explicitly here.
 */
static inline void dequant_block_q4_0(const block_q4_0 *blk, float *out) {
    const float d = fp16_to_fp32(blk->d);
    for (int j = 0; j < QK4_0 / 2; j++) {
        const int lo = (blk->qs[j] & 0x0F) - 8;
        const int hi = (blk->qs[j] >>   4) - 8;
        out[j]              = lo * d;
        out[j + QK4_0 / 2]  = hi * d;
    }
}

/* Dot product of one dequantized Q4_0 row (n_cols elems) against x. */
static float row_dot_q4_0(const uint8_t *row_data, const float *x, int n_cols) {
    const int n_blocks = n_cols / QK4_0;
    const block_q4_0 *blocks = (const block_q4_0 *)row_data;
    float acc = 0.0f;
    float buf[QK4_0];

    for (int b = 0; b < n_blocks; b++) {
        dequant_block_q4_0(&blocks[b], buf);
        const float *xp = x + (size_t)b * QK4_0;
        float local = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            local += buf[j] * xp[j];
        }
        acc += local;
    }
    return acc;
}

static void gemv_q4_0(const uint8_t *W, const float *x, float *y,
                       int n_rows, int n_cols) {
    const size_t row_bytes = (size_t)(n_cols / QK4_0) * BLOCK_BYTES;
    for (int i = 0; i < n_rows; i++) {
        const uint8_t *row = W + (size_t)i * row_bytes;
        y[i] = row_dot_q4_0(row, x, n_cols);
    }
}

static uint8_t *read_file(const char *path, size_t expected_bytes, size_t *out_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "ERROR: %s is empty or unreadable\n", path);
        exit(1);
    }
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) {
        fprintf(stderr, "ERROR: out of memory reading %s\n", path);
        exit(1);
    }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) {
        fprintf(stderr, "ERROR: short read on %s (got %zu of %ld bytes)\n", path, rd, sz);
        exit(1);
    }
    if (expected_bytes && (size_t)sz != expected_bytes) {
        fprintf(stderr,
            "WARNING: %s is %ld bytes, expected %zu — shape/args may be wrong.\n",
            path, sz, expected_bytes);
    }
    if (out_bytes) *out_bytes = (size_t)sz;
    return buf;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s W.bin x.bin y_ref.bin [n_rows=2048] [n_cols=11008] [iterations=200]\n",
            argv[0]);
        return 1;
    }

    const char *w_path = argv[1];
    const char *x_path = argv[2];
    const char *y_path = argv[3];
    int n_rows = (argc > 4) ? atoi(argv[4]) : 2048;
    int n_cols = (argc > 5) ? atoi(argv[5]) : 11008;
    int iters  = (argc > 6) ? atoi(argv[6]) : 200;

    if (n_cols % QK4_0 != 0) {
        fprintf(stderr, "ERROR: n_cols (%d) must be a multiple of %d for Q4_0\n", n_cols, QK4_0);
        return 1;
    }
    if (iters < 10) {
        fprintf(stderr, "WARNING: iterations=%d is low; median/percentile stats will be noisy.\n", iters);
    }

    size_t expected_w_bytes = (size_t)n_rows * (n_cols / QK4_0) * BLOCK_BYTES;
    size_t expected_x_bytes = (size_t)n_cols * sizeof(float);
    size_t expected_y_bytes = (size_t)n_rows * sizeof(float);

    size_t w_bytes, x_bytes, y_bytes;
    uint8_t *W     = read_file(w_path, expected_w_bytes, &w_bytes);
    uint8_t *x_raw = read_file(x_path, expected_x_bytes, &x_bytes);
    uint8_t *y_raw = read_file(y_path, expected_y_bytes, &y_bytes);

    float *x     = (float *)x_raw;
    float *y_ref = (float *)y_raw;
    float *y_out = malloc(sizeof(float) * (size_t)n_rows);
    if (!y_out) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }

    printf("=== Q4_0 GEMV kernel benchmark ===\n");
    printf("Weight file : %s (%zu bytes)\n", w_path, w_bytes);
    printf("Input file  : %s (%zu bytes, %d floats)\n", x_path, x_bytes, n_cols);
    printf("Ref file    : %s (%zu bytes, %d floats)\n", y_path, y_bytes, n_rows);
    printf("Shape       : y[%d] = W[%d,%d] @ x[%d]\n", n_rows, n_rows, n_cols, n_cols);
    printf("Iterations  : %d\n\n", iters);

    /* ---------------- correctness check (single run) ---------------- */
    gemv_q4_0(W, x, y_out, n_rows, n_cols);

    double max_abs_diff = 0.0, sum_abs_diff = 0.0;
    double dot = 0.0, norm_ref = 0.0, norm_out = 0.0;
    for (int i = 0; i < n_rows; i++) {
        double diff = fabs((double)y_out[i] - (double)y_ref[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        sum_abs_diff += diff;
        dot      += (double)y_out[i] * (double)y_ref[i];
        norm_ref += (double)y_ref[i] * (double)y_ref[i];
        norm_out += (double)y_out[i] * (double)y_out[i];
    }
    double mean_abs_diff = sum_abs_diff / n_rows;
    double cosine_sim = dot / (sqrt(norm_ref) * sqrt(norm_out) + 1e-12);

    /*
     * y_ref was produced by ggml's own Q4_0 kernel on the identical
     * quantized weights, so the only expected source of divergence is
     * floating-point summation-order drift (11008-term sums, different
     * blocking/threading order). Real bugs — wrong nibble order, wrong
     * row stride, endianness — tend to show up as either near-zero
     * cosine similarity or wildly-off magnitude, not small drift, so
     * the thresholds below are intentionally tight.
     */
    const double TOL_COSINE  = 0.999;
    const double TOL_MAXDIFF = 0.05;

    printf("--- Correctness ---\n");
    printf("Max abs diff   : %.6f\n", max_abs_diff);
    printf("Mean abs diff  : %.6f\n", mean_abs_diff);
    printf("Cosine similarity vs reference: %.8f\n", cosine_sim);

    int pass = (cosine_sim >= TOL_COSINE) && (max_abs_diff <= TOL_MAXDIFF);
    printf("Result: %s (require cosine_sim >= %.3f and max_abs_diff <= %.3f)\n\n",
           pass ? "PASS" : "FAIL / INSPECT", TOL_COSINE, TOL_MAXDIFF);

    if (!pass) {
        fprintf(stderr,
            "WARNING: output does not closely match the reference. This is more likely\n"
            "a real bug (block layout, row stride, endianness, wrong n_rows/n_cols) than\n"
            "ordinary float summation drift -- inspect before trusting the timing numbers\n"
            "below.\n\n");
    }

    /* ---------------------------- timing ---------------------------- */
    double *times = malloc(sizeof(double) * (size_t)iters);
    if (!times) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }

    for (int w = 0; w < 5; w++) gemv_q4_0(W, x, y_out, n_rows, n_cols); /* warmup */

    for (int it = 0; it < iters; it++) {
        double t0 = now_sec();
        gemv_q4_0(W, x, y_out, n_rows, n_cols);
        double t1 = now_sec();
        times[it] = t1 - t0;
    }

    qsort(times, (size_t)iters, sizeof(double), cmp_double);
    double median = times[iters / 2];
    double p10 = times[(int)(iters * 0.1)];
    double p90 = times[(int)(iters * 0.9)];
    double mean = 0.0;
    for (int i = 0; i < iters; i++) mean += times[i];
    mean /= iters;

    /* GOPS convention: multiply + add counted as 2 ops, matching the
     * convention used elsewhere in this project's roofline estimates. */
    double total_macs   = (double)n_rows * (double)n_cols;
    double gops_median  = (2.0 * total_macs) / median / 1e9;
    double gops_mean    = (2.0 * total_macs) / mean   / 1e9;

    printf("--- Timing (%d iterations, 5 warmup runs excluded) ---\n", iters);
    printf("Median latency : %.4f ms\n", median * 1000.0);
    printf("Mean latency   : %.4f ms\n", mean   * 1000.0);
    printf("P10 / P90      : %.4f ms / %.4f ms\n", p10 * 1000.0, p90 * 1000.0);
    printf("\n--- Derived throughput (this kernel only, single-threaded) ---\n");
    printf("MACs per call  : %.0f (%.3f MMAC)\n", total_macs, total_macs / 1e6);
    printf("GOPS (median)  : %.3f\n", gops_median);
    printf("GOPS (mean)    : %.3f\n", gops_mean);

    free(times);
    free(W);
    free(x_raw);
    free(y_raw);
    free(y_out);
    return pass ? 0 : 2;
}

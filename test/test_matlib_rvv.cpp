#include <cstdio>
#include <cmath>

#include "matlib.h"
#include "matlib_cpu.h"
#include "test_rvv.h"

extern "C" {

int main() {

    enable_vector_operations();

    uint32_t seed = 0xdeadbeef;
    uint64_t start, total;
    srand(seed);

    // array gen
    scalar_t *A = alloc_array_2d(N, O);
    scalar_t *B = alloc_array_2d(M, O);
    scalar_t *f = alloc_array_1d(N * M);
    gen_rand_2d(A, N, O);
    gen_rand_2d(B, M, O);
    gen_rand_1d(f, N * M);

    printf("matmul:         ");
    scalar_t *golden = alloc_array_2d(N, M);
    scalar_t *actual = alloc_array_2d(N, M);
    matmul_cpu(A, B, golden, N, M, O);
    start = read_cycles();
    matmul(A, B, actual, N, M, O);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden, actual, N, M) ? "pass" : "fail", total);

    matset(actual, 0.0, N, M);
    printf("matmult:        ");
    start = read_cycles();
    matmul(A, B, actual, N, M, O, 8);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden, actual, N, M) ? "pass" : "fail", total);

    matset(actual, 0.0, N, M);
    printf("matmuli:        ");
    int *ID = (int *)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) ID[i] = i * O;
    start = read_cycles();
    matmul(A, B, actual, N, M, O, 8, ID);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden, actual, N, M) ? "pass" : "fail", total);

    // print_array_2d(A, N, O, "float32", "A");
    // print_array_2d(B, M, O, "float32", "B");
    // print_array_2d(golden, N, M, "float32", "golden");
    // print_array_2d(actual, N, M, "float32", "actual");

    // array gen
    scalar_t *G = alloc_array_2d(N, M);
    scalar_t *H = alloc_array_2d(M, N);
    scalar_t *V = alloc_array_2d(1, M);
    scalar_t *W = alloc_array_2d(M, 1);
    gen_rand_2d(G, N, M);
    gen_rand_2d(H, M, N);
    gen_rand_2d(V, 1, M);
    gen_rand_2d(W, M, 1);

    printf("matvec:         ");
    scalar_t *golden_vec = alloc_array_2d(N, 1);
    scalar_t *actual_vec = alloc_array_2d(N, 1);
    matvec_cpu(G, V, golden_vec, N, M);
    start = read_cycles();
    matvec(G, V, actual_vec, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden_vec, actual_vec, N, 1) ? "pass" : "fail", total);

    printf("matvec_t:       ");
    matvec_transpose_cpu(H, W, golden_vec, M, N);
    start = read_cycles();
    matvec_transpose(H, W, actual_vec, M, N);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(golden_vec, actual_vec, N, 1) ? "pass" : "fail", total);

    // array gen
    A = alloc_array_2d(N, M);
    B = alloc_array_2d(N, M);
    scalar_t *C = alloc_array_2d(N, M);
    scalar_t *D = alloc_array_2d(N, M);
    gen_rand_2d(A, N, M);
    gen_rand_2d(B, N, M);

    printf("maxcoeff:       ");
    scalar_t max_cpu = maxcoeff_cpu(A, N, M);
    start = read_cycles();
    scalar_t max_actual = maxcoeff(A, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", is_equal(max_cpu, max_actual, 1e-6) ? "pass" : "fail", total);

    printf("mincoeff:       ");
    scalar_t min_cpu = mincoeff_cpu(A, N, M);
    start = read_cycles();
    scalar_t min_actual = mincoeff(A, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", is_equal(min_cpu, min_actual, 1e-6) ? "pass" : "fail", total);

    printf("matmulf:        ");
    matmulf_cpu(A, C, 10.0f, N, M);
    start = read_cycles();
    matmulf(A, D, 10.0f, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matsub:         ");
    matsub_cpu(A, B, C, N, M);
    start = read_cycles();
    matsub(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matadd:         ");
    matadd_cpu(A, B, C, N, M);
    start = read_cycles();
    matadd(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matneg:         ");
    matneg_cpu(A, C, N, M);
    start = read_cycles();
    matneg(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matcopy:        ");
    matcopy_cpu(A, C, N, M);
    start = read_cycles();
    matcopy(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwiseabs:       ");
    cwiseabs_cpu(A, C, N, M);
    start = read_cycles();
    cwiseabs(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwisemin:       ");
    cwisemin_cpu(A, B, C, N, M);
    start = read_cycles();
    cwisemin(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwisemax:       ");
    cwisemax_cpu(A, B, C, N, M);
    start = read_cycles();
    cwisemax(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("cwisemul:       ");
    cwisemul_cpu(A, B, C, N, M);
    start = read_cycles();
    cwisemul(A, B, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matset:         ");
    matset_cpu(C, 5.0, N, M);
    start = read_cycles();
    matset(D, 5.0, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matsetv:        ");
    matsetv_cpu(A, f, N, M);
    start = read_cycles();
    matsetv(A, f, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, N, M) ? "pass" : "fail", total);

    printf("matnorm:        ");
    scalar_t norm_cpu = matnorm_cpu(A, N, M);
    start = read_cycles();
    scalar_t norm_actual = matnorm(A, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", is_equal(norm_cpu, norm_actual, 1e-6) ? "pass" : "fail", total);

    C = alloc_array_2d(M, N);
    D = alloc_array_2d(M, N);

    printf("transpose:      ");
    transpose_cpu(A, C, N, M);
    start = read_cycles();
    transpose(A, D, N, M);
    total = read_cycles() - start;
    printf("%s (%lu)\n", compare_2d(C, D, M, N) ? "pass" : "fail", total);

done:
    return(0);
}

}

#include <cstdio>
#include <cmath>

#include "matlib.h"
#include "test_rvv.h"

extern "C" {
uint64_t start, total;

inline void test_matmul() {
    scalar_t *A = alloc_array_2d(N, O);
    scalar_t *B = alloc_array_2d(M, O);
    gen_rand_2d(A, N, O);
    gen_rand_2d(B, M, O);
    printf("matmul:         ");
    scalar_t *actual = alloc_array_2d(N, M);
    start = read_cycles();
    matmul(A, B, actual, N, M, O);
    total = read_cycles() - start;
    printf("%lu\n", total);
    printf("matmult:        ");
    start = read_cycles();
    matmul(A, B, actual, N, M, O, 8);
    total = read_cycles() - start;
    printf("%lu\n", total);
    matset(actual, 0.0, N, M);
    printf("matmuli:        ");
    int *ID = (int *)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) ID[i] = i * O;
    start = read_cycles();
    matmul(A, B, actual, N, M, O, 8, ID);
    total = read_cycles() - start;
    printf("%lu\n", total);
}

inline void test_matvec() {
    scalar_t *G = alloc_array_2d(N, M);
    scalar_t *V = alloc_array_2d(1, M);
    gen_rand_2d(G, N, M);
    gen_rand_2d(V, 1, M);
    printf("matvec:         ");
    scalar_t *actual_vec = alloc_array_2d(N, 1);
    start = read_cycles();
    matvec(G, V, actual_vec, N, M);
    total = read_cycles() - start;
    printf("%lu\n", total);
}

int main() {

    enable_vector_operations();

    uint32_t seed = 0xdeadbeef;
    srand(seed);

    test_matmul();
    test_matvec();

done:
    return(0);
}

}

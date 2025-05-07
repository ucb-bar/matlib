#pragma once
#ifndef TINYMPC_MATLIB_RVV_H
#define TINYMPC_MATLIB_RVV_H

#include <cstdio>
#include <climits>
#include <cmath>

#include "riscv_vector.h"

extern "C"
{

// RVV-specific function declarations
int maxcoeff_rvv(int *a, int n, int m);
int mincoeff_rvv(int *a, int n, int m);
int matnorm_rvv(int *a, int n, int m);
void matneg_rvv(int *a, int *b, int n, int m);
void cwiseabs_rvv(int *a, int *b, int n, int m);
void cwisemin_rvv(int *a, int *b, int *c, int n, int m);
void cwisemax_rvv(int *a, int *b, int *c, int n, int m);
void cwisemul_rvv(int *a, int *b, int *c, int n, int m);
void matmul_rvv(int *a, int *b, int *c, int n, int m, int o, int tile_size, int *ind_a);
void matmul_rvvt(int *a, int *b, int *c, int i, int j, int k, int n, int m, int o, int tile_size, int *ind_a);
void matvec_rvv(int *a, int *b, int *c, int n, int m);
void matvec_transpose_rvv(int *a, int *b, int *c, int n, int m);
void matmulf_rvv(int *a, int *b, int f, int n, int m);
void matsub_rvv(int *a, int *b, int *c, int n, int m);
void matadd_rvv(int *a, int *b, int *c, int n, int m);
void transpose_rvv(int *a, int *b, int n, int m);
void matcopy_rvv(const int *a, int *b, int n, int m);
void matset_rvv(int *a, int f, int n, int m);
void matsetv_rvv(int *a, int *f, int n, int m);

#define maxcoeff maxcoeff_rvv
#define mincoeff mincoeff_rvv
#define matnorm matnorm_rvv
#define matneg matneg_rvv
#define cwiseabs cwiseabs_rvv
#define cwisemin cwisemin_rvv
#define cwisemax cwisemax_rvv
#define cwisemul cwisemul_rvv
#define matmul matmul_rvv
#define matvec matvec_rvv
#define matvec_transpose matvec_transpose_rvv
#define matmulf matmulf_rvv
#define matsub matsub_rvv
#define matadd matadd_rvv
#define transpose transpose_rvv
#define matcopy matcopy_rvv
#define matset matset_rvv
#define matsetv matsetv_rvv

#define BATCH 4


// matrix maximum coefficient
inline int maxcoeff_rvv(int *ptr_a, int n, int m) {
    int max = std::numeric_limits<int>::min();
    vint32m1_t vec_max = __riscv_vmv_s_x_i32m1(max, 1);
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vec_max = __riscv_vredmax_vs_i32_i32(vec_a, vec_max, vl);
    }
    max = __riscv_vmv_x_s_i32m1_i32(vec_max);
    return max;
}

// matrix min coefficient
inline int mincoeff_rvv(int *ptr_a, int n, int m) {
    int min = std::numeric_limits<int>::max();
    vint32m1_t vec_min = __riscv_vmv_s_x_i32m1(min, 1);
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vec_min = __riscv_vredmin_vs_i32_i32(vec_a, vec_min, vl);
    }
    min = __riscv_vmv_x_s_i32m1_i32(vec_min);
    return min;
}

// matrix unary negative
inline void matneg_rvv(int *ptr_a, int *ptr_b, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_b = __riscv_vneg_v_i32(vec_a, vl);
        __riscv_vse32_v_i32(ptr_b + l, vec_b, vl);
    }
}

// matrix l2 norm
inline int matnorm_rvv(int *ptr_a, int n, int m) {
    int k = m * n, l = 0;
    size_t vlmax = __riscv_vsetvlmax_e32();
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
    vint32_t vec_s = __riscv_vmv_v_x_i32(0, vlmax);
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vec_s = __riscv_vmacc_vv_i32(vec_s, vec_a, vec_a, vl);
    }
    vint32m1_t vec_sum = __riscv_vredsum_vs_i32_i32(vec_s, vec_zero, vlmax);
    int sum = __riscv_vmv_x_s_i32m1_i32(vec_sum);
    return sqrt(sum);
}

// matrix coefficient-wise abs
inline void cwiseabs_rvv(int *ptr_a, int *ptr_b, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_n = __riscv_vneg_v_i32(vec_a, vl);
        vint32_t vec_b = __riscv_vmax_vv_i32(vec_a, vec_n, vl);
        __riscv_vse32_v_i32(ptr_b + l, vec_b, vl);
    }
}

// matrix coefficient-wise min
inline void cwisemin_rvv(int *ptr_a, int *ptr_b, int *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_b = __riscv_vle32_v_i32(ptr_b + l, vl);
        vint32_t vec_c = __riscv_vmin_vv_i32(vec_a, vec_b, vl);
        __riscv_vse32_v_i32(ptr_c + l, vec_c, vl);
    }
}

// matrix coefficient-wise multiplication
inline void cwisemul_rvv(int *ptr_a, int *ptr_b, int *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_b = __riscv_vle32_v_i32(ptr_b + l, vl);
        vint32_t vec_c = __riscv_vmul_vv_i32(vec_a, vec_b, vl);
        __riscv_vse32_v_i32(ptr_c + l, vec_c, vl);
    }
}

// matrix coefficient-wise max
inline void cwisemax_rvv(int *ptr_a, int *ptr_b, int *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_b = __riscv_vle32_v_i32(ptr_b + l, vl);
        vint32_t vec_c = __riscv_vmax_vv_i32(vec_a, vec_b, vl);
        __riscv_vse32_v_i32(ptr_c + l, vec_c, vl);
    }
}

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
inline void matmul_rvv(int *a, int *b, int *c, int n, int m, int o, int tile_size = -1, int *ind_a = 0) {
    if (tile_size == -1) {
        size_t vlmax = __riscv_vsetvlmax_e32();
        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int *ptr_a = a + i * o; // row major
                int *ptr_b = b + j * o; // column major
                int k = o;
                vint32_t vec_s = __riscv_vmv_v_x_i32(0, vlmax);
                for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
                    vl = __riscv_vsetvl_e32(k);
                    vint32_t vec_a = __riscv_vle32_v_i32(ptr_a, vl);
                    vint32_t vec_b = __riscv_vle32_v_i32(ptr_b, vl);
                    vec_s = __riscv_vmacc_vv_i32(vec_s, vec_a, vec_b, vl);
                }
                vint32m1_t vec_sum = __riscv_vredsum_vs_i32_i32(vec_s, vec_zero, vlmax);
                int sum = __riscv_vmv_x_s_i32m1_i32(vec_sum);
                c[i * m + j] = sum;
            }
        }
    } else {
        // matset_rvv(c, 0.0f, n, m);
        for (int i = 0; i < n; i += tile_size) {
            for (int j = 0; j < m; j += tile_size) {
                for (int k = 0; k < o; k += tile_size) {
                    // printf("i: %d j: %d k: %d n: %d m: %d o: %d\n", i, j, k, n, m, o);
                    matmul_rvvt(a, b, c, i, j, k, n, m, o, tile_size, ind_a);
                }
            }
        }
    }
}

inline void matmul_rvvt(int *a, int *b, int *c, int i, int j, int k, int n, int m, int o, int tile_size, int *ind_a) {
    vint32m1_t v1, v2, v3, v4, v5, v6, v7, v8;
    vint32m1_t *vec_r[8] = { &v1, &v2, &v3, &v4, &v5, &v6, &v7, &v8 };
    int *A = a + (ind_a ? ind_a[i] : i * o) + k;
    int *B = b + j * o + k;
    int *C = c + i * m + j;
    int N = i + tile_size <= n ? tile_size : n % tile_size;
    int M = j + tile_size <= m ? tile_size : m % tile_size;
    int O = k + tile_size <= o ? tile_size : o % tile_size;
    // printf("A: %d B: %d C: %d N: %d M: %d O: %d\n", A, B, C, N, M, O);
    size_t vlmax = __riscv_vsetvlmax_e32();
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
    for (int I = 0; I < N; ++I) {
        for (int J = 0; J < M; J += BATCH) {
            int P = J + BATCH < M ? BATCH : M - J;
            int *ptr_a = A + I * o; // row major
            int *ptr_b = B + J * o; // column major
            int K = O;
            for (int L = 0; L < P; L++) {
                *(vec_r[L]) = __riscv_vmv_v_x_i32(0, vlmax);
            }
            for (size_t vl; K > 0; K -= vl, ptr_a += vl, ptr_b += vl) {
                vl = __riscv_vsetvl_e32(K);
                // printf("I: %d J: %d K: %d N: %d M: %d O: %d ptr_a: %x ptr_b: %x vl: %d\n", I, J, K, N, M, O, ptr_a, ptr_b, vl);
                vint32_t vec_a = __riscv_vle32_v_i32(ptr_a, vl);
                for (int L = 0; L < P; L++) {
                    vint32_t vec_b = __riscv_vle32_v_i32(ptr_b + L * o, vl);
                    *(vec_r[L]) = __riscv_vmacc_vv_i32(*(vec_r[L]), vec_a, vec_b, vl);
                }
            }
            for (int L = 0; L < P; L++) {
                vint32m1_t vec_sum = __riscv_vredsum_vs_i32_i32(*(vec_r[L]), vec_zero, vlmax);
                int sum = __riscv_vmv_x_s_i32m1_i32(vec_sum);
                C[I * m + J + L] = k == 0 ? sum : C[I * m + J + L] + sum;
            }
        }
    }
}

/*  a is row major
 *        j
 *    1 2 3 4     9
 *  i 5 6 7 8  *  6
 *    9 8 7 6     5 j
 *                4
 */
inline void matvec_rvv(int *a, int *b, int *c, int n, int m) {
    size_t vlmax = __riscv_vsetvlmax_e32();
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
    for (int i = 0; i < n; ++i) {
        int k = m;
        int *ptr_a = a + i * m; // row major
        int *ptr_b = b;
        vint32_t vec_s = __riscv_vmv_v_x_i32(0, vlmax);
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e32(k);
            vint32_t vec_a = __riscv_vle32_v_i32(ptr_a, vl);
            vint32_t vec_b = __riscv_vle32_v_i32(ptr_b, vl);
            vec_s = __riscv_vmacc_vv_i32(vec_s, vec_a, vec_b, vl);
        }
        vint32m1_t vec_sum = __riscv_vredsum_vs_i32_i32(vec_s, vec_zero, vlmax);
        int sum = __riscv_vmv_x_s_i32m1_i32(vec_sum);
        c[i] = sum;
    }
}

/*  a is col major
 *      j         i
 *  9 6 5 4  *  1 5 9
 *              2 6 8
 *              3 7 7 j
 *              4 8 6
 */
inline void matvec_transpose_rvv(int *a, int *b, int *c, int n, int m) {
    size_t vlmax = __riscv_vsetvlmax_e32();
    vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
    for (int i = 0; i < m; ++i) {
        int k = n;
        int *ptr_a = a + i; // column major
        int *ptr_b = b;
        vint32_t vec_s = __riscv_vmv_v_x_i32(0, vlmax);
        for (size_t vl; k > 0; k -= vl, ptr_a += m * vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e32(k);
            vint32_t vec_a = __riscv_vlse32_v_i32(ptr_a, m * sizeof(int), vl);
            vint32_t vec_b = __riscv_vle32_v_i32(ptr_b, vl);
            vec_s = __riscv_vmacc_vv_i32(vec_s, vec_a, vec_b, vl);
        }
        vint32m1_t vec_sum = __riscv_vredsum_vs_i32_i32(vec_s, vec_zero, vlmax);
        int sum = __riscv_vmv_x_s_i32m1_i32(vec_sum);
        c[i] = sum;
    }
}

// matrix scalar multiplication
inline void matmulf_rvv(int *ptr_a, int *ptr_b, int f, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_b = __riscv_vmul_vx_i32(vec_a, f, vl);
        __riscv_vse32_v_i32(ptr_b + l, vec_b, vl);
    }
}

// matrix subtraction
inline void matsub_rvv(int *ptr_a, int *ptr_b, int *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_b = __riscv_vle32_v_i32(ptr_b + l, vl);
        vint32_t vec_c = __riscv_vsub_vv_i32(vec_a, vec_b, vl);
        __riscv_vse32_v_i32(ptr_c + l, vec_c, vl);
    }
}

// matrix addition
inline void matadd_rvv(int *ptr_a, int *ptr_b, int *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        vint32_t vec_b = __riscv_vle32_v_i32(ptr_b + l, vl);
        vint32_t vec_c = __riscv_vadd_vv_i32(vec_a, vec_b, vl);
        __riscv_vse32_v_i32(ptr_c + l, vec_c, vl);
    }
}

// matrix transpose
inline void transpose_rvv(int *a, int *b, int n, int m) {
    for (int j = 0; j < m; ++j) {
        int *ptr_a = a + j;
        int *ptr_b = b + j * n;
        int k = n;
        int l = 0;
        for (size_t vl; k > 0; k -= vl, l += vl, ptr_a = a + l * m + j, ptr_b += vl) {
            vl = __riscv_vsetvl_e32(k);
            vint32_t vec_a = __riscv_vlse32_v_i32(ptr_a, sizeof(int) * m, vl);
            __riscv_vse32(ptr_b, vec_a, vl);
        }
    }
};

// matrix copy
inline void matcopy_rvv(const int *ptr_a, int *ptr_b, int n, int m) {
    int k = n * m, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vle32_v_i32(ptr_a + l, vl);
        __riscv_vse32_v_i32(ptr_b + l, vec_a, vl);
    }
}

inline void matset_rvv(int *ptr_a, int f, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_a = __riscv_vmv_v_x_i32(f, vl);
        __riscv_vse32_v_i32(ptr_a + l, vec_a, vl);
    }
}

inline void matsetv_rvv(int *ptr_a, int *f, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e32(k);
        vint32_t vec_f = __riscv_vle32_v_i32(f + l, vl);;
        __riscv_vse32_v_i32(ptr_a + l, vec_f, vl);
    }
}

}
#endif //TINYMPC_MATLIB_RVV_H

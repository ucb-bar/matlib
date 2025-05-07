#
# Created by widyadewi on 4/27/24.
#

import re
import sys
import traceback
from textwrap import *

indents = ''

class Unroller:

    lmul = 1
    sew = 32
    vlen = 512
    batch = 4
    vlmax = int(lmul * vlen / sew)
    idx = 1

    @classmethod
    def print(cls, indents, line):
        print(indent(dedent(line), indents))

    @classmethod
    def setlmul(cls, indents, lmul):
        Unroller.lmul = lmul

    @classmethod
    def matadd(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1}, vec_{i+2};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};
                float *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_f32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vfadd_vv_f32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_f32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def matsub(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1}, vec_{i+2};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};
                float *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_f32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vfsub_vv_f32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_f32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def maxcoeff(cls, indents, target, tail, a, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                float max = std::numeric_limits<float>::min();
                vfloat32m1_t vec_max = __riscv_vfmv_s_f_f32m1(max, 1);
                vfloat32_t vec_{i};
                float *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_max = __riscv_vfredmax_vs_f32_f32(vec_{i}, vec_max, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.print(indents, f"""\
                max = __riscv_vfmv_f_s_f32m1_f32(vec_max);
                {target} = max {tail};\
            """)
        Unroller.idx += 1

    @classmethod
    def mincoeff(cls, indents, target, tail, a, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                float min = std::numeric_limits<float>::max();
                vfloat32m1_t vec_min = __riscv_vfmv_s_f_f32m1(min, 1);
                vfloat32_t vec_{i};
                float *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_min = __riscv_vfredmin_vs_f32_f32(vec_{i}, vec_min, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.print(indents, f"""\
                min = __riscv_vfmv_f_s_f32m1_f32(vec_min);
                {target} = min {tail};\
            """)
        Unroller.idx += 1

    @classmethod
    def matnorm(cls, indents, target, tail, a, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i};
                float *ptr_{i} = {a};
                size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                vfloat32m1_t vec_zero_{i} = __riscv_vfmv_v_f_f32m1(0, vlmax_{i});
                vfloat32_t vec_s_{i} = __riscv_vfmv_v_f_f32(0, vlmax_{i});\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_s_{i} = __riscv_vfmacc_vv_f32(vec_s_{i}, vec_{i}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.print(indents, f"""\
                vfloat32m1_t vec_sum_{i} = __riscv_vfredusum_vs_f32_f32(vec_s_{i}, vec_zero_{i}, vlmax_{i});
                float sum_{i} = __riscv_vfmv_f_s_f32m1_f32(vec_sum_{i});
                {target} = sqrt(sum_{i}) {tail};\
            """)
        Unroller.idx += 1

    @classmethod
    def matneg(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vfneg_v_f32(vec_{i}, {vl});
                __riscv_vse32_v_f32(ptr_{i+1} + {l}, vec_{i+1}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2

    @classmethod
    def transpose(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i};
                float *ptr_{i}, *ptr_{i+1};\
                """)
        for j in range(m):
            k = n
            l = 0
            Unroller.print(indents, f"""\
                ptr_{i} = {a} + {j};
                ptr_{i+1} = {b} + {j * n};\
                """)
            while k > 0:
                vl = min(k, Unroller.vlmax)
                k -= vl
                l += vl
                Unroller.print(indents, f"""\
                    vec_{i} = __riscv_vlse32_v_f32(ptr_{i}, sizeof(float) * {m}, {vl});
                    __riscv_vse32(ptr_{i+1}, vec_{i}, {vl});
                    ptr_{i} = {a} + {l * m + j};
                    ptr_{i+1} += {vl};\
                """)
        Unroller.idx += 2

    @classmethod
    def cwiseabs(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vfabs_v_f32(vec_{i}, {vl});
                __riscv_vse32_v_f32(ptr_{i+1} + {l}, vec_{i+1}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2

    @classmethod
    def cwisemin(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1}, vec_{i+2};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};
                float *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_f32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vfmin_vv_f32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_f32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def cwisemax(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1}, vec_{i+2};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};
                float *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_f32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vfmax_vv_f32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_f32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def matmul(cls, indents, a, b, c, n, m, o, tile_size="-1", ind_a=""):
        n = eval(n)
        m = eval(m)
        o = eval(o)
        t = eval(tile_size)
        i = Unroller.idx
        nt = n % t
        mt = m % t
        ot = o % t
        if t != -1:
            def matmul_tile(N, M, O, reset=False):
                indents28 = indents + " " * 28
                Unroller.print(indents28, f"""\
                                // n {n} m {m} o {o} N {N} M {M} O {O}\
                                """)
                for I in range(N):
                    ptr_a = f"{a} + {ind_a}[i + {I}] + k" if ind_a else f"{a} + (i + {I}) * {o} + k"
                    for J in range(0, M, Unroller.batch):
                        P = Unroller.batch if J + Unroller.batch < M else M - J;
                        Unroller.print(indents28, f"""\
                                ptr_{i} = {ptr_a};
                                ptr_{i+1} = B_{i} + {J * o};\
                                """)
                        k = O
                        for L in range(P):
                            Unroller.print(indents28, f"""\
                                vec_r_{i}_{L} = __riscv_vfmv_v_f_f32(0, vlmax_{i});\
                                """)
                        while k > 0:
                            vl = min(k, Unroller.vlmax)
                            Unroller.print(indents28, f"""\
                                vec_{i} = __riscv_vle32_v_f32(ptr_{i}, {vl});\
                                """)
                            for L in range(P):
                                Unroller.print(indents28, f"""\
                                vec_{i+1} = __riscv_vle32_v_f32(ptr_{i+1} + {L * o}, {vl});
                                vec_r_{i}_{L} = __riscv_vfmacc_vv_f32(vec_r_{i}_{L}, vec_{i}, vec_{i+1}, {vl});\
                                """)
                            if k - vl > 0:
                                Unroller.print(indents28, f"""\
                                ptr_{i} += {vl};
                                ptr_{i+1} += {vl};\
                                """)
                            k -= vl
                        for L in range(P):
                            Unroller.print(indents28, f"""\
                                vec_sum_{i} = __riscv_vfredusum_vs_f32_f32(vec_r_{i}_{L}, vec_zero_{i}, vlmax_{i});
                                C_{i}[{I * m + J + L}] { '=' if reset else '+=' } __riscv_vfmv_f_s_f32m1_f32(vec_sum_{i});\
                                """)

            def matmul_helper(reset):
                Unroller.print(indents + 16 * " ", f"""\
                                if (a_full_{i}) {{
                                    if (b_full_{i}) {{
                                        if (c_full_{i}) {{ """)
                matmul_tile(t, t, t, reset)
                Unroller.print(indents + 24 * " ", f"""\
                                        }} else {{ """)
                matmul_tile(t, t, ot, reset)
                Unroller.print(indents + 20 * " ", f"""\
                                        }}
                                    }} else {{
                                        if (c_full_{i}) {{ """)
                matmul_tile(t, mt, t, reset)
                Unroller.print(indents + 24 * " ", f"""\
                                        }} else {{ """)
                matmul_tile(t, mt, ot, reset)
                Unroller.print(indents + 16 * " ", f"""\
                                        }}
                                    }}
                                }} else {{
                                    if (b_full_{i}) {{
                                        if (c_full_{i}) {{ """)
                matmul_tile(nt, t, t, reset)
                Unroller.print(indents + 24 * " ", f"""\
                                        }} else {{ """)
                matmul_tile(nt, t, ot, reset)
                Unroller.print(indents + 20 * " ", f"""\
                                        }}
                                    }} else {{
                                        if (c_full_{i}) {{ """)
                matmul_tile(nt, mt, t, reset)
                Unroller.print(indents + 24 * " ", f"""\
                                        }} else {{ """)
                matmul_tile(nt, mt, ot, reset)
                Unroller.print(indents + 16 * " ", f"""\
                                        }}
                                    }}
                                }}""")

            for L in range(Unroller.batch):    
                Unroller.print(indents, f"""\
                vfloat32_t vec_r_{i}_{L};\
                """)
            Unroller.print(indents, f"""\
                vfloat32_t vec_s_{i}, vec_{i}, vec_{i+1}, vec_{i+2};
                float *ptr_{i}; float *ptr_{i+1}; float *ptr_{i+2} = {c};
                size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                vfloat32m1_t vec_sum_{i};
                vfloat32m1_t vec_zero_{i} = __riscv_vfmv_v_f_f32m1(0, vlmax_{i});\
                """)
            Unroller.print(indents, f"""\
                int nt_{i} = {n} % {t};
                int mt_{i} = {m} % {t};
                int ot_{i} = {o} % {t};                
                for (int i = 0; i < {n}; i += {t}) {{
                    for (int j = 0; j < {m}; j += {t}) {{
                        for (int k = 0; k < {o}; k += {t}) {{
                            float *B_{i} = {b} + j * {o} + k;
                            float *C_{i} = {c} + i * {m} + j;
                            bool a_full_{i} = (i + {t}) < {n};
                            bool b_full_{i} = (j + {t}) < {m};
                            bool c_full_{i} = (k + {t}) < {o};
                            int N_{i} = a_full_{i} ? {t} : nt_{i};
                            int M_{i} = b_full_{i} ? {t} : mt_{i};
                            int O_{i} = c_full_{i} ? {t} : ot_{i};
                            if (k == 0) {{ """)
            matmul_helper(True)
            Unroller.print(indents + 12 * " ", f"""\
                            }} else {{ """)
            matmul_helper(False)
            Unroller.print(indents, f"""\
                            }}
                        }}
                    }}
                }} """)
            Unroller.idx += 3
        else:
            Unroller.print(indents, f"""\
                            vfloat32_t vec_s_{i}, vec_{i}, vec_{i+1}, vec_{i+2};
                            float *ptr_{i}; float *ptr_{i+1}; float *ptr_{i+2} = {c};
                            size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                            vfloat32m1_t vec_sum_{i};
                            vfloat32m1_t vec_zero_{i} = __riscv_vfmv_v_f_f32m1(0, vlmax_{i});\
                            """)
            for I in range(n):
                for J in range(m):
                    k = o
                    Unroller.print(indents, f"""\
                            ptr_{i} = {a} + {I * o};
                            ptr_{i+1} = {b} + {J * o};
                            vec_s_{i} = __riscv_vfmv_v_f_f32(0, vlmax_{i});\
                            """)
                    while k > 0:
                        vl = min(k, Unroller.vlmax)
                        Unroller.print(indents, f"""\
                            vec_{i} = __riscv_vle32_v_f32(ptr_{i}, {vl});
                            vec_{i+1} = __riscv_vle32_v_f32(ptr_{i+1}, {vl});
                            vec_s_{i} = __riscv_vfmacc_vv_f32(vec_s_{i}, vec_{i}, vec_{i+1}, {vl});\
                            """)
                        if k - vl > 0:
                            Unroller.print(indents, f"""\
                            ptr_{i} += {vl};
                            ptr_{i+1} += {vl};\
                            """)
                        k -= vl
                    Unroller.print(indents, f"""\
                            vec_sum_{i} = __riscv_vfredusum_vs_f32_f32(vec_s_{i}, vec_zero_{i}, vlmax_{i});
                            ptr_{i+2}[{I * m + J}] = __riscv_vfmv_f_s_f32m1_f32(vec_sum_{i});\
                            """)
            Unroller.idx += 3

    @classmethod
    def matvec(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        i = Unroller.idx
        Unroller.print(indents, f"""\
                    vfloat32_t vec_s_{i}, vec_{i}, vec_{i+1}, vec_{i+2};
                    float *ptr_{i}; float *ptr_{i+1}; float *ptr_{i+2} = {c};
                    size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                    vfloat32m1_t vec_sum_{i};
                    vfloat32m1_t vec_zero_{i} = __riscv_vfmv_v_f_f32m1(0, vlmax_{i});\
                    """)
        k = n
        I = 0
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                    ptr_{i} = {a} + {I * m};
                    ptr_{i+1} = {b};
                    vec_s_{i} = __riscv_vfmv_v_f_f32(0, vlmax_{i});\
                    """)
            for J in range(m):
                Unroller.print(indents, f"""\
                    vec_{i} = __riscv_vlse32_v_f32(ptr_{i} + {J}, {m} * sizeof(float), {vl});
                    vec_s_{i} = __riscv_vfmacc_vf_f32(vec_s_{i}, *(ptr_{i+1} + {J}), vec_{i}, {vl});\
                    """)
            Unroller.print(indents, f"""\
                    __riscv_vse32_v_f32(ptr_{i+2} + {I}, vec_s_{i}, {vl});\
                    """)
            k -= vl
            I += vl
        Unroller.idx += 3

    @classmethod
    def matvec_transpose(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        i = Unroller.idx
        Unroller.print(indents, f"""\
                    vfloat32_t vec_s_{i}, vec_{i}, vec_{i+1}, vec_{i+2};
                    float *ptr_{i}; float *ptr_{i+1}; float *ptr_{i+2} = {c};
                    size_t vlmax_{i} = __riscv_vsetvlmax_e32();
                    vfloat32m1_t vec_sum_{i};
                    vfloat32m1_t vec_zero_{i} = __riscv_vfmv_v_f_f32m1(0, vlmax_{i});\
                    """)
        k = m
        I = 0
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                    ptr_{i} = {a} + {I};
                    ptr_{i+1} = {b};
                    vec_s_{i} = __riscv_vfmv_v_f_f32(0, vlmax_{i});\
                    """)
            for J in range(n):
                Unroller.print(indents, f"""\
                    vec_{i} =  __riscv_vle32_v_f32(ptr_{i} + {J * m}, {vl});
                    vec_s_{i} = __riscv_vfmacc_vf_f32(vec_s_{i}, *(ptr_{i+1} + {J}), vec_{i}, {vl});\
                    """)
            Unroller.print(indents, f"""\
                    __riscv_vse32_v_f32(ptr_{i+2} + {I}, vec_s_{i}, {vl});\
                    """)
            k -= vl
            I += vl
        Unroller.idx += 3

    @classmethod
    def cwisemul(cls, indents, a, b, c, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1}, vec_{i+2};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};
                float *ptr_{i+2} = {c};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vle32_v_f32(ptr_{i+1} + {l}, {vl});
                vec_{i+2} = __riscv_vfmul_vv_f32(vec_{i}, vec_{i+1}, {vl});
                __riscv_vse32_v_f32(ptr_{i+2} + {l}, vec_{i+2}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 3

    @classmethod
    def matmulf(cls, indents, a, b, f, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i}, vec_{i+1};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                vec_{i+1} = __riscv_vfmul_vf_f32(vec_{i}, {f}, {vl});
                __riscv_vse32_v_f32(ptr_{i+1} + {l}, vec_{i+1}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2

    @classmethod
    def matset(cls, indents, a, f, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i};
                float *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vfmv_v_f_f32({f}, {vl});
                __riscv_vse32_v_f32(ptr_{i} + {l}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 1

    @classmethod
    def matsetv(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i};
                float *ptr_{i} = {a};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32({b} + {l}, {vl});
                __riscv_vse32_v_f32(ptr_{i} + {l}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 1

    @classmethod
    def matcopy(cls, indents, a, b, n, m):
        n = eval(n)
        m = eval(m)
        k = m * n
        l = 0
        i = Unroller.idx
        Unroller.print(indents, f"""\
                vfloat32_t vec_{i};
                float *ptr_{i} = {a};
                float *ptr_{i+1} = {b};\
            """)
        while k > 0:
            vl = min(k, Unroller.vlmax)
            Unroller.print(indents, f"""\
                vec_{i} = __riscv_vle32_v_f32(ptr_{i} + {l}, {vl});
                __riscv_vse32_v_f32(ptr_{i+1} + {l}, vec_{i}, {vl});\
            """)
            k -= vl
            l += vl
        Unroller.idx += 2


if __name__ == "__main__":

    DEBUG = False
    prefix = "__rvvu__"
    input_file = sys.argv[1]
    Unroller.lmul = sys.argv[2]

    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    print('#include "matlib_lmul.h"')

    for line_no, line in enumerate(lines):

        if re.match(r"\# (\d+) .+", line):
            continue

        pattern_void = r"(\s+){}(\w+)\s*\((.+)\)".format(prefix)
        match_void = re.match(pattern_void, line)
        pattern_return = r"(\s+)(.+)\s*=\s*{}(.+?)\s*\((.+)\)(.*)".format(prefix)
        match_return = re.match(pattern_return, line)
        if match_void:
            indents = match_void.group(1)
            method = match_void.group(2)
            arguments = match_void.group(3).strip().split(', ')
            if DEBUG:
                print("indents: ", len(indents))
                print("function: ", method)
                print("arguments: ", arguments)
            try:
                print(f"{indents}// line {line_no}: {dedent(line)}", end="")
                getattr(Unroller, method)(indents, *arguments)
            except Exception as e: 
                if DEBUG: print(f"// ERROR: {e}", end="")
                print(f"{indents}{method}_rvv({match_void.group(3)});")
        elif match_return:
            indents = match_return.group(1)
            target = match_return.group(2)
            method = match_return.group(3)
            arguments = match_return.group(4).strip().split(', ')
            tail = match_return.group(5)
            try:
                print(f"{indents}// line {line_no}: {dedent(line)}", end="")
                getattr(Unroller, method)(indents, target, tail, *arguments)
            except Exception as e:
                print(f"{indents}{target} = {method}_rvvu({match_return.group(4)});")
        else:
            print(line, end="")



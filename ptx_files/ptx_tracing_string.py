# =============================================================================
# ptx_files/ptx_tracing_string.py -- PTX opcode -> address-formula builders
# =============================================================================
# locality_guru.py imports this module for its op_dict (ADD, MADLO, SHL, ...):
# each function turns the operands of one PTX instruction into a Python
# expression string, which locality_map_coalescing.py later evaluates per
# thread.
#
# The copy committed to LocalityGuru_2 (4228045) is not this module: it is a
# copy of ptx_tracing_main.py that ends with an empty `if __name__ ==
# "__main__":`, so `python3 locality_guru.py` stops with an IndentationError.
# These functions were reconstructed instruction by instruction from the
# compiled module committed next to it,
# ptx_files/__pycache__/ptx_tracing_string.cpython-38.pyc (source lines 14-112),
# and behave exactly like the compiled originals, including their quirks
# (SQRT calls pow() with one argument, CLZ loops while a % 10 is non-zero),
# except for two fixes in MADLO/MAD (mad.lo d, a, b, c = low 32 bits of a*b,
# plus c):
#   * the mask was 0x0000000011111111, which keeps only bits 0, 4, 8, ... of
#     each nibble pair instead of the low 32 bits; it is now 0xFFFFFFFF;
#   * the result is now parenthesised.  Without the parentheses an enclosing
#     MUL/SHL bound to the last term only: mul.wide.s32 %rd, %r(mad), 4 gave
#     "a*b + c*4" instead of "(a*b + c)*4".
# These formulas feed the coalescing statistics of locality_map_coalescing.py.
# The locality matrices no longer use them (see ptx_exec.py).
# The rest of that module (OPERATE, tracing, file_open and the per-benchmark
# *_init test drivers) is not used by locality_guru.py and is not included.
# =============================================================================
import os, sys
import json
import math

param_dict = dict()
tidx = 1
tidy = 1
ctaidy = 16

tidx_list = list(range(0, 16))
tidy_list = list(range(0, 16))
ctaidy_list = list(range(0, ctaidy))
ctaid_map = list()


def ADD(a, b):
    return f"({a} + {b})"


def ABS(a):
    if a < 0:
        a *= -1
    return a


def SUB(a, b):
    return f"({a} - {b})"


def MADLO(a, b, c):
    return f"((( {a} *  {b} )&0xFFFFFFFF) + {c})"


def MAD(a, b, c):
    return f"((( {a} *  {b} )&0xFFFFFFFF) + {c})"


def MUL(a, b):
    return f"({a} * {b})"


def LD(a):
    return f"{a}"


def ST(a):
    return f"{a}"


def CVTA(a):
    return f"{a}"


def CVT(a):
    return f"{a}"


def MOV(a):
    return f"{a}"


def NOT(a):
    return f"~{a}"


def SHR(a, b):
    return f"({a} >> {b})"


def SHL(a, b):
    return f"({a} << {b})"


def OR(a, b):
    return f"({a} | {b})"


def AND(a, b):
    return f"({a} & {b})"


def DIV(a, b):
    return f"({a} / {b})"


def SELP(a, b, c):
    if c == 1:
        return f"{a}"
    else:
        return f"{b}"


def SETP(a, b, c):
    return SELP(a, b, c)


def SETP_GE(a, b, c):
    if c >= 1:
        return f"{a}"
    else:
        return f"{b}"


def SETP_LT(a, b, c):
    if c < 1:
        return f"{a}"
    else:
        return f"{b}"


def SETP_NE(a, b, c):
    if c != 1:
        return f"{a}"
    else:
        return f"{b}"


def SETP_GT(a, b, c):
    if c > 1:
        return f"{a}"
    return f"{b}"


def SETP_EQ(a, b, c):
    if c == 1:
        return f"{a}"
    return f"{b}"


def BFE(a, b, c):
    return f"{a}"


def PRMT(a, b, c):
    return f"{a}"


def SQRT(a):
    return f"{pow(a)}"


def MIN(a, b):
    if a > b:
        return f"{b}"
    return f"{a}"


def MAX(a, b):
    if a < b:
        return f"{b}"
    return f"{a}"


def NEG(a):
    return f"{a * -1}"


def CLZ(a):
    cnt = 0
    while a % 10:
        cnt += 1
        continue
        break
    return f"{cnt}"


def RCP(a):
    return f"(1/{a})"


def REM(a):
    return f"{a}"

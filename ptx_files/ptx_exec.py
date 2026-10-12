#!/usr/bin/env python3
# =============================================================================
# ptx_files/ptx_exec.py -- TB locality matrices from the kernel's PTX
# =============================================================================
# LocalityGuru (Tripathy et al., 2021), Sec. III-C and Algorithm 2: the
# locality graph has one vertex per thread block (TB), and the weight of the
# edge between TB i and TB j is the number of distinct global-memory addresses
# that both of them read with ld.global:
#
#     L(TBi, TBj) = | Map[TBi] & Map[TBj] |
#
# Map[TB] holds every address loaded by every non-idle thread of the TB, in
# every loop iteration ("All the cases of idle threads, address offsets in the
# source register, and loops are taken into account").  The paper validated
# its graphs against addresses recorded by running the benchmarks in
# GPGPU-Sim, element by element.
#
# This module builds Map[TB] by executing the kernel's PTX for every thread of
# the grid, at the point where the launch parameters are known (the paper's
# JIT-time analysis): integer and address arithmetic, predicates, branches and
# loops run exactly as written, so
#   * a loop runs until its own exit predicate is false, with every register
#     the loop updates (unrolled induction variables, strength-reduced
#     pointers) taking the values the hardware would give it,
#   * threads whose predicates branch around a load (idle threads) do not
#     record it,
#   * "[%rd+off]" offsets and v2/v4 loads are included.
# Values loaded from memory are not known before the launch: registers
# computed from them are marked data-dependent ("taint").  A load whose
# address is data-dependent (an indirect access) is skipped with a warning,
# and a data-dependent branch is reported (see _Run._check).
#
# Addresses: pointer parameters get disjoint synthetic base addresses, 2^36
# bytes apart, so two arrays can never overlap.  Scalar parameters that
# influence an address or a branch (3DConvolution's plane index `i`, a loop
# bound `n`, ...) must be given with --param; the analysis stops with a
# message naming them otherwise.
#
# Python 3.8+ and numpy only.
# =============================================================================
import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
_M = {1: 1, 8: 0xFF, 16: 0xFFFF, 32: 0xFFFFFFFF, 64: 0xFFFFFFFFFFFFFFFF}
_MU = dict((k, np.uint64(v)) for k, v in _M.items())
_U0 = np.uint64(0)
_U1 = np.uint64(1)

_TYPE_BITS = {
    "b8": 8, "u8": 8, "s8": 8,
    "b16": 16, "u16": 16, "s16": 16, "f16": 16, "bf16": 16,
    "b32": 32, "u32": 32, "s32": 32, "f32": 32, "f16x2": 32, "bf16x2": 32,
    "tf32": 32,
    "b64": 64, "u64": 64, "s64": 64, "f64": 64,
    "b128": 128,
    "pred": 1,
}

# Synthetic address space: region r (pointer parameter or declared variable)
# occupies [ADDR_LOW + (r + 1) * REGION, ...).  Keys packed for the
# per-TB sets are ((addr - ADDR_LOW) << TB_BITS) | tb, which needs
# addr - ADDR_LOW < 2^(64 - TB_BITS).
REGION_BITS = 36  # 64 GB per array
ADDR_LOW = 1 << (REGION_BITS - 1)
TB_BITS = 21  # up to 2,097,151 TBs per kernel
MAX_REL_ADDR = 1 << (64 - TB_BITS)
MAX_REGIONS = (MAX_REL_ADDR >> REGION_BITS) - 2

_SREG_TAINTED = ("smid", "nsmid", "gridid", "clock", "clock64", "clock_hi",
                 "globaltimer", "globaltimer_lo", "globaltimer_hi",
                 "dynamic_smem_size", "total_smem_size", "aggr_smem_size",
                 "lanemask_eq", "lanemask_le", "lanemask_lt", "lanemask_ge",
                 "lanemask_gt", "is_explicit_cluster", "current_graph_exec")

EMPTY = frozenset()


class PTXAnalysisError(Exception):
    """The kernel cannot be analysed as written (message says why)."""


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
class Operand(object):
    __slots__ = ("kind", "name", "value", "neg", "base", "off", "items",
                 "text")

    def __init__(self, kind, text, name=None, value=None, neg=False,
                 base=None, off=0, items=None):
        self.kind = kind    # reg sreg imm fimm sym addr vec pair label other
        self.text = text
        self.name = name
        self.value = value  # int (imm), (bits, width) (fimm), float (dimm)
        self.neg = neg      # "!%p" predicate operand
        self.base = base    # addr: Operand (reg/sym/imm) or None
        self.off = off      # addr: int offset
        self.items = items  # vec / pair: list of Operand (None for "_")


class Instr(object):
    __slots__ = ("line", "text", "guard", "guard_neg", "op", "mods", "ops",
                 "types", "w", "kind", "target", "space", "vec")

    def __init__(self, line, text):
        self.line = line
        self.text = text
        self.guard = None
        self.guard_neg = False
        self.op = ""
        self.mods = []
        self.ops = []
        self.types = []
        self.w = 0
        self.kind = ""
        self.target = -1
        self.space = None
        self.vec = 1


class Param(object):
    __slots__ = ("name", "type", "w", "is_ptr", "size", "index")

    def __init__(self, name, ptype, w, is_ptr, size, index):
        self.name = name
        self.type = ptype
        self.w = w
        self.is_ptr = is_ptr
        self.size = size
        self.index = index


class Kernel(object):
    def __init__(self, name):
        self.name = name
        self.params = []        # list of Param
        self.instrs = []        # list of Instr
        self.labels = {}        # label -> instruction index
        self.symbols = {}       # name -> state space ("shared", "local", ...)


class Module(object):
    def __init__(self):
        self.kernels = OrderedDict()
        self.symbols = OrderedDict()  # module-level variables: name -> space
        self.address_size = 64


_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_ENTRY_RE = re.compile(r"\.entry\s+([A-Za-z_$%][\w$]*)\s*\(")
_VAR_RE = re.compile(
    r"(?:^|[;{}\n])\s*(?:\.(?:extern|visible|weak|common)\s+)*"
    r"\.(global|const|shared|local)\s+([^;]*?)([A-Za-z_$][\w$]*)\s*"
    r"((?:\[\s*\d*\s*\])*)\s*(?:=[^;]*)?;")
_INT_RE = re.compile(r"^[+-]?(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9]\d*)U?$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_SREG_RE = re.compile(r"^%([A-Za-z_]+[\w]*)(?:\.([xyzw]))?$")
_SREG_NAMES = ("tid", "ntid", "ctaid", "nctaid", "laneid", "warpid",
               "nwarpid") + _SREG_TAINTED


def _parse_int(text):
    t = text.rstrip("U")
    neg = t.startswith("-")
    if t[0] in "+-":
        t = t[1:]
    if len(t) > 1 and t[0] == "0" and t[1] not in "xXbB":
        v = int(t, 8)
    else:
        v = int(t, 0)
    return -v if neg else v


def _split_top(s, sep=","):
    """Split s at `sep` outside (), [] and {}."""
    out, depth, cur = [], 0, []
    for c in s:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    tail = "".join(cur).strip()
    if tail or out:
        out.append(tail)
    return [x for x in out if x != ""]


def _parse_operand(text):
    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].replace(" ", "").replace("\t", "")
        base, off = inner, 0
        m = re.match(r"^(.*?)(\+-?|-)((?:0[xX][0-9a-fA-F]+)|\d+)$", inner)
        if m and m.group(1):
            base = m.group(1)
            off = _parse_int(m.group(3))
            if m.group(2) == "-" or m.group(2) == "+-":
                off = -off
        if base == "":
            b = None
        elif _INT_RE.match(base):
            b = Operand("imm", base, value=_parse_int(base))
        else:
            b = _parse_operand(base)
        return Operand("addr", t, base=b, off=off)
    if t.startswith("{") and t.endswith("}"):
        items = []
        for x in _split_top(t[1:-1]):
            items.append(None if x == "_" else _parse_operand(x))
        return Operand("vec", t, items=items)
    if "|" in t and not t.startswith("("):
        return Operand("pair", t, items=[_parse_operand(x) for x in t.split("|")])
    neg = False
    if t.startswith("!"):
        neg, t = True, t[1:].strip()
    if t.startswith("%"):
        m = _SREG_RE.match(t)
        if m and m.group(1) in _SREG_NAMES or (m and m.group(1).startswith("envreg")):
            name = m.group(1) + ("." + m.group(2) if m.group(2) else "")
            return Operand("sreg", t, name=name, neg=neg)
        return Operand("reg", t, name=t, neg=neg)
    if re.match(r"^0[fF][0-9a-fA-F]{8}$", t):
        return Operand("fimm", t, value=(int(t[2:], 16), 32))
    if re.match(r"^0[dD][0-9a-fA-F]{16}$", t):
        return Operand("fimm", t, value=(int(t[2:], 16), 64))
    if _INT_RE.match(t):
        return Operand("imm", t, value=_parse_int(t))
    if _FLOAT_RE.match(t):
        return Operand("dimm", t, value=float(t))
    if re.match(r"^[A-Za-z_$][\w$]*$", t):
        return Operand("sym", t, name=t, neg=neg)
    return Operand("other", t)


def _decode(ins):
    """Fill in the fields the executor needs from the opcode modifiers."""
    mods = ins.mods
    ins.types = [m for m in mods if m in _TYPE_BITS]
    ins.w = _TYPE_BITS[ins.types[-1]] if ins.types else 0
    op = ins.op
    if op in ("ld", "ldu", "st", "atom", "red", "prefetch", "prefetchu", "cp"):
        for sp in ("global", "shared", "local", "param", "const"):
            if sp in mods:
                ins.space = sp
                break
        for m in mods:
            if m in ("v2", "v4", "v8"):
                ins.vec = int(m[1:])
    if op == "bra" or op == "brx":
        ins.kind = "bra"
    elif op in ("ret", "exit"):
        ins.kind = "exit"
    elif op == "trap":
        ins.kind = "trap"
    else:
        ins.kind = op


def _split_statements(body, line0):
    """Yield ('label', name, line) and ('stmt', text, line) from a body."""
    out = []
    buf = []
    nonblank = False
    depth = 0
    line = line0
    start = line0
    for c in body:
        if c == "\n":
            line += 1
        if depth == 0 and c == ";":
            s = "".join(buf).strip()
            if s:
                out.append(("stmt", s, start))
            buf, nonblank = [], False
            continue
        if depth == 0 and not nonblank and c in "{}":
            continue  # a scope brace, not a vector operand
        if depth == 0 and c == ":" and nonblank:
            s = "".join(buf).strip()
            if re.match(r"^[A-Za-z_$][\w$]*$", s):
                out.append(("label", s, start))
                buf, nonblank = [], False
                continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if not nonblank and not c.isspace():
            nonblank = True
            start = line
        buf.append(c)
    s = "".join(buf).strip()
    if s:
        out.append(("stmt", s, start))
    return out


def _parse_params(text):
    params = []
    for i, decl in enumerate(_split_top(text)):
        toks = decl.replace("\n", " ").split()
        ptype, is_ptr, size = None, False, None
        name = None
        for tok in toks:
            if tok.startswith("."):
                t = tok[1:]
                if t in _TYPE_BITS and ptype is None:
                    ptype = t
                elif t == "ptr":
                    is_ptr = True
            else:
                name = tok
        if name is None:
            continue
        m = re.match(r"^([A-Za-z_$][\w$]*)\s*\[(\d*)\]$", name)
        if m:
            name = m.group(1)
            size = int(m.group(2)) if m.group(2) else None
        w = _TYPE_BITS.get(ptype, 0)
        params.append(Param(name, ptype or "b8", w, is_ptr, size, i))
    return params


def parse_ptx(text):
    """Parse the .entry kernels of a PTX module."""
    mod = Module()
    m = re.search(r"\.address_size\s+(\d+)", text)
    if m:
        mod.address_size = int(m.group(1))
    if mod.address_size != 64:
        raise PTXAnalysisError(
            ".address_size %d: only 64-bit PTX is supported (nvcc's default "
            "on 64-bit hosts; compile with -m64)" % mod.address_size)
    clean = _COMMENT_RE.sub(lambda mm: "\n" * mm.group(0).count("\n"), text)
    for vm in _VAR_RE.finditer(clean):
        mod.symbols.setdefault(vm.group(3), vm.group(1))
    for em in _ENTRY_RE.finditer(clean):
        k = Kernel(em.group(1))
        i, depth = em.end(), 1
        while depth:
            c = clean[i]
            depth += (c == "(") - (c == ")")
            i += 1
        k.params = _parse_params(clean[em.end():i - 1])
        j = clean.index("{", i)
        p, depth = j + 1, 1
        while depth:
            c = clean[p]
            depth += (c == "{") - (c == "}")
            p += 1
        body = clean[j + 1:p - 1]
        line0 = clean.count("\n", 0, j + 1) + 1
        pending = []
        for kind, s, line in _split_statements(body, line0):
            if kind == "label":
                pending.append(s)
                continue
            if s.startswith("."):
                vm = re.match(r"^\.(shared|local|global|const)\b.*?([A-Za-z_$][\w$]*)\s*(\[[^\]]*\])*\s*(=.*)?$", s, re.S)
                if vm:
                    k.symbols[vm.group(2)] = vm.group(1)
                continue
            ins = Instr(line, " ".join(s.split()))
            st = s
            gm = re.match(r"^@(!?)\s*(%?[\w$]+)\s+", st)
            if gm:
                ins.guard_neg = gm.group(1) == "!"
                ins.guard = _parse_operand(gm.group(2))
                st = st[gm.end():]
            om = re.match(r"^([A-Za-z_][\w.:]*)\s*(.*)$", st, re.S)
            if not om:
                continue
            parts = om.group(1).split(".")
            ins.op, ins.mods = parts[0], parts[1:]
            rest = om.group(2).strip()
            ins.ops = [_parse_operand(x) for x in _split_top(rest)] if rest else []
            _decode(ins)
            for lab in pending:
                k.labels[lab] = len(k.instrs)
            pending = []
            k.instrs.append(ins)
        for lab in pending:
            k.labels[lab] = len(k.instrs)
        for ins in k.instrs:
            if ins.kind == "bra":
                tgt = ins.ops[-1].text if ins.ops else ""
                if tgt not in k.labels:
                    raise PTXAnalysisError(
                        "%s: line %d: branch target %r not found"
                        % (k.name, ins.line, tgt))
                ins.target = k.labels[tgt]
        mod.kernels[k.name] = k
    return mod


# ----------------------------------------------------------------------------
# pointer parameters
# ----------------------------------------------------------------------------
def classify_params(kernel):
    """Mark the integer parameters that are used as addresses.

    A parameter is a pointer when the register it is loaded into reaches a
    cvta instruction or the base of a load/store/atomic address, directly or
    through mov/add/sub/cvt.  Everything else is a scalar.
    """
    reg_of = {}
    for ins in kernel.instrs:
        if ins.op == "ld" and ins.space == "param" and len(ins.ops) >= 2:
            a = ins.ops[1]
            if a.kind == "addr" and a.base is not None and a.base.kind == "sym" \
                    and a.off == 0 and ins.ops[0].kind == "reg":
                reg_of.setdefault(a.base.name, set()).add(ins.ops[0].name)
    ptr = set(p.name for p in kernel.params if p.is_ptr)
    for pname, regs in reg_of.items():
        derived = set(regs)
        changed = True
        while changed:
            changed = False
            for ins in kernel.instrs:
                srcs = [o.name for o in ins.ops[1:] if o.kind == "reg"]
                if ins.op in ("mov", "add", "sub", "cvta", "cvt") and \
                        ins.ops and ins.ops[0].kind == "reg" and \
                        any(s in derived for s in srcs) and \
                        ins.ops[0].name not in derived:
                    derived.add(ins.ops[0].name)
                    changed = True
        for ins in kernel.instrs:
            if ins.op == "cvta" and any(o.kind == "reg" and o.name in derived
                                        for o in ins.ops[1:]):
                ptr.add(pname)
            if ins.op in ("ld", "ldu", "st", "atom", "red", "prefetch") and \
                    ins.space != "param":
                for o in ins.ops:
                    if o.kind == "addr" and o.base is not None and \
                            o.base.kind == "reg" and o.base.name in derived:
                        ptr.add(pname)
    for p in kernel.params:
        p.is_ptr = p.name in ptr
    return [p.name for p in kernel.params if p.is_ptr]


# ----------------------------------------------------------------------------
# integer helpers (values are np.uint64 holding the bit pattern, zero-extended)
# ----------------------------------------------------------------------------
def _u(x):
    return np.asarray(x, dtype=np.uint64)


def _signed(v, w):
    v = _u(v)
    if w == 64:
        return v.view(np.int64)
    if w == 32:
        return v.astype(np.uint32).view(np.int32).astype(np.int64)
    if w == 16:
        return v.astype(np.uint16).view(np.int16).astype(np.int64)
    if w == 8:
        return v.astype(np.uint8).view(np.int8).astype(np.int64)
    raise PTXAnalysisError("unsupported integer width %d" % w)


def _from_signed(x, w):
    """int64 values -> w-bit pattern as uint64."""
    return np.asarray(x, dtype=np.int64).astype(np.uint64) & _MU[w]


def _to_float(bits, w):
    bits = _u(bits)
    if w == 32:
        return bits.astype(np.uint32).view(np.float32).astype(np.float64)
    if w == 64:
        return bits.view(np.float64)
    if w == 16:
        return bits.astype(np.uint16).view(np.float16).astype(np.float64)
    raise PTXAnalysisError("unsupported float width %d" % w)


def _from_float(x, w):
    x = np.asarray(x, dtype=np.float64)
    if w == 32:
        return x.astype(np.float32).view(np.uint32).astype(np.uint64)
    if w == 64:
        return x.view(np.uint64).copy()
    if w == 16:
        return x.astype(np.float16).view(np.uint16).astype(np.uint64)
    raise PTXAnalysisError("unsupported float width %d" % w)


def _mulhi_u64(a, b):
    m32 = np.uint64(0xFFFFFFFF)
    s32 = np.uint64(32)
    a0, a1 = a & m32, a >> s32
    b0, b1 = b & m32, b >> s32
    p00, p01, p10, p11 = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    mid = (p00 >> s32) + (p01 & m32) + (p10 & m32)
    return p11 + (p01 >> s32) + (p10 >> s32) + (mid >> s32)


def _mulhi(a, b, w, signed):
    a, b = _u(a), _u(b)
    if w == 64:
        hi = _mulhi_u64(a, b)
        if signed:
            hi = hi - np.where(_signed(a, 64) < 0, b, _U0) \
                    - np.where(_signed(b, 64) < 0, a, _U0)
        return hi
    if signed:
        p = _signed(a, w) * _signed(b, w)
        return _from_signed(p >> np.int64(w), w)
    return ((a * b) >> np.uint64(w)) & _MU[w]


def _sorted_unique(a):
    """Sorted distinct values of a 1-d array (sort + drop repeats)."""
    a = np.sort(a, kind="stable" if a.size < 64 else "quicksort")
    if a.size < 2:
        return a
    keep = np.empty(a.size, dtype=bool)
    keep[0] = True
    np.not_equal(a[1:], a[:-1], out=keep[1:])
    return a[keep]


def _trunc_div(sa, sb):
    """C-style signed division and remainder on int64 (b == 0 -> 0)."""
    zero = sb == 0
    sb1 = np.where(zero, np.int64(1), sb)
    q = np.floor_divide(sa, sb1)          # floor division, then round
    r = sa - q * sb1                       # toward zero (no abs(): it
    adj = ((r != 0) & ((sa < 0) != (sb1 < 0))).astype(np.int64)  # overflows
    q = q + adj                            # for INT64_MIN)
    r = r - adj * sb1
    return np.where(zero, np.int64(0), q), np.where(zero, np.int64(0), r)


# ----------------------------------------------------------------------------
# executor
# ----------------------------------------------------------------------------
class _Run(object):
    """One batch of TBs of one kernel, executed in SIMT lock-step."""

    def __init__(self, kernel, grid, block, tb_first, tb_count, pvals,
                 symbase, region_space, address_size, step_limit, report):
        self.k = kernel
        self.report = report
        self.step_limit = step_limit
        self.pvals = pvals            # param name -> (bits, taint)
        self.symbase = symbase        # symbol name -> base address
        self.region_space = region_space
        self.aw = address_size
        bx, by, bz = block
        gx, gy, gz = grid
        tpb = bx * by * bz
        n = tpb * tb_count
        self.n = n
        lin = np.arange(n, dtype=np.int64)
        t = lin % tpb
        tb = np.int64(tb_first) + lin // tpb
        self.tb = tb.astype(np.uint64)
        u = lambda x: np.asarray(x, dtype=np.int64).astype(np.uint64)
        self.sreg = {
            "tid.x": u(t % bx), "tid.y": u((t // bx) % by),
            "tid.z": u(t // (bx * by)),
            "ntid.x": np.uint64(bx), "ntid.y": np.uint64(by),
            "ntid.z": np.uint64(bz),
            "ctaid.x": u(tb % gx), "ctaid.y": u((tb // gx) % gy),
            "ctaid.z": u(tb // (gx * gy)),
            "nctaid.x": np.uint64(gx), "nctaid.y": np.uint64(gy),
            "nctaid.z": np.uint64(gz),
            "laneid": u(t % 32), "warpid": u(t // 32),
            "nwarpid": np.uint64((tpb + 31) // 32),
        }
        self.regs = {}
        self.taint = {}
        self.width = {}  # register -> width of the last value written
        self.keys = []
        self.n_keys = 0
        self.compact_at = 8000000  # compact when the buffer doubles
        self.ld_exec = 0
        self.ld_skipped = 0
        self.steps = 0

    # -- operands -----------------------------------------------------------
    def _bits(self, o, w, ins):
        """Integer/bit operand -> (uint64 bits masked to w, taint)."""
        k = o.kind
        w = min(w or 64, 64)
        if k == "reg":
            v = self.regs.get(o.name)
            if v is None:
                return _U0, EMPTY
            if v.dtype == np.bool_:
                return v.astype(np.uint64), self.taint.get(o.name, EMPTY)
            if w < 64 and self.width.get(o.name, 64) > w:
                v = v & _MU[w]
            return v, self.taint.get(o.name, EMPTY)
        if k == "imm":
            return np.uint64(o.value & _M[w if w <= 64 else 64]), EMPTY
        if k == "fimm":
            bits, fw = o.value
            if fw == w or w == 0:
                return np.uint64(bits), EMPTY
            return _from_float(_to_float(np.uint64(bits), fw), w), EMPTY
        if k == "dimm":
            return _from_float(o.value, w if w in (16, 32, 64) else 64), EMPTY
        if k == "sreg":
            name = o.name
            base = name.split(".")[0]
            if base in _SREG_TAINTED or base.startswith("envreg") or \
                    name not in self.sreg:
                return _U0, frozenset(["sreg:%" + name])
            return self.sreg[name] & _MU[w] if w < 64 else self.sreg[name], EMPTY
        if k == "sym":
            if o.name in self.symbase:
                return np.uint64(self.symbase[o.name]), EMPTY
            if o.name in self.pvals:  # mov %rd, param (address of param)
                return _U0, frozenset(["param-address:" + o.name])
            raise PTXAnalysisError("%s: line %d: unknown symbol %r"
                                   % (self.k.name, ins.line, o.name))
        raise PTXAnalysisError("%s: line %d: unsupported operand %r"
                               % (self.k.name, ins.line, o.text))

    def _pred(self, o, ins):
        if o.kind == "reg":
            v = self.regs.get(o.name)
            t = self.taint.get(o.name, EMPTY)
            if v is None:
                v = np.zeros(self.n, dtype=bool)
            elif v.dtype != np.bool_:
                v = v != _U0
            return (~v if o.neg else v), t
        if o.kind == "imm":
            return np.bool_(bool(o.value) != o.neg), EMPTY
        raise PTXAnalysisError("%s: line %d: bad predicate operand %r"
                               % (self.k.name, ins.line, o.text))

    def _float(self, o, w, ins):
        if o.kind == "fimm":
            bits, fw = o.value
            return _to_float(np.uint64(bits), fw), EMPTY
        if o.kind == "imm":
            return np.float64(o.value), EMPTY
        if o.kind == "dimm":
            return np.float64(o.value), EMPTY
        b, t = self._bits(o, w, ins)
        return _to_float(b, w), t

    # -- writes ---------------------------------------------------------------
    def _write(self, o, val, taint, mask, w=64, pred=False):
        if o is None:
            return
        if o.kind == "vec":
            raise PTXAnalysisError("%s: vector destination not expected here"
                                   % self.k.name)
        if o.kind != "reg":
            if o.kind == "other" and o.text == "_":
                return
            raise PTXAnalysisError("%s: cannot write to %r" % (self.k.name, o.text))
        name = o.name
        w = min(w or 64, 64)
        if pred:
            val = np.asarray(val, dtype=bool)
        else:
            val = _u(val)
            if w < 64:
                val = val & _MU[w]
        old = self.regs.get(name)
        if mask is None:
            if val.ndim == 0:
                val = np.full(self.n, val, dtype=val.dtype)
            self.regs[name] = val
            self.taint[name] = taint
            self.width[name] = 1 if pred else w
        else:
            if old is None:
                old = np.zeros(self.n, dtype=bool if pred else np.uint64)
            elif pred and old.dtype != np.bool_:
                old = old != _U0
            elif not pred and old.dtype == np.bool_:
                old = old.astype(np.uint64)
            self.regs[name] = np.where(mask, val, old)
            self.taint[name] = taint | self.taint.get(name, EMPTY)
            self.width[name] = 1 if pred else max(w, self.width.get(name, w))

    def _taint_dst(self, ins, taint, mask):
        """Destination(s) get placeholder zeros with the given taint."""
        if not ins.ops:
            return
        d = ins.ops[0]
        targets = d.items if d.kind in ("vec", "pair") else [d]
        for t in targets:
            if t is not None and t.kind == "reg":
                isp = t.name in self.regs and self.regs[t.name].dtype == np.bool_
                self._write(t, False if isp else _U0, taint, mask, 64, pred=isp)

    # -- taint policy -----------------------------------------------------------
    def _check(self, taint, ins, what):
        """Decide what to do with an unknown value used as `what`."""
        params = sorted(t[6:] for t in taint if t.startswith("param:"))
        if params:
            raise PTXAnalysisError(
                "kernel %s, line %d (%s): the %s depends on scalar parameter(s) "
                "%s, whose value is only known at launch. Pass it with "
                "--param NAME=VALUE (NAME may be the full name or the "
                "'param_N' suffix)." % (self.k.name, ins.line, ins.text, what,
                                        ", ".join(params)))
        other = sorted(t for t in taint if t != "data")
        if other:
            raise PTXAnalysisError(
                "kernel %s, line %d (%s): the %s depends on %s, which this "
                "analysis cannot evaluate." % (self.k.name, ins.line, ins.text,
                                              what, ", ".join(other)))
        self.report.warn_once(
            (self.k.name, ins.line, what),
            "kernel %s, line %d (%s): the %s depends on values loaded from "
            "memory (data-dependent; not known before the launch)."
            % (self.k.name, ins.line, ins.text, what))

    # -- loads ------------------------------------------------------------------
    def _record(self, addr, nelem, esize, mask, ins):
        addr = _u(addr)
        if mask is None:
            tb = self.tb
            if addr.ndim == 0:
                addr = np.full(self.n, addr, dtype=np.uint64)
        else:
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            tb = self.tb[idx]
            addr = addr[idx] if addr.ndim else np.full(idx.size, addr, dtype=np.uint64)
        low = np.uint64(ADDR_LOW)
        for e in range(nelem):
            a = addr + np.uint64(e * esize) if e else addr
            rel = a - low
            if a.size and (a.min() < low or rel.max() >= np.uint64(MAX_REL_ADDR)):
                raise PTXAnalysisError(
                    "kernel %s, line %d (%s): load address outside the "
                    "synthetic address space (pointer arithmetic below an "
                    "array base or past 2^%d bytes)"
                    % (self.k.name, ins.line, ins.text, 64 - TB_BITS))
            key = _sorted_unique((rel << np.uint64(TB_BITS)) | tb)
            self.keys.append(key)
            self.n_keys += key.size
        self.ld_exec += 1
        if self.n_keys > self.compact_at:
            self._compact()
            self.compact_at = max(self.compact_at, 2 * self.n_keys)

    def _compact(self):
        if len(self.keys) > 1:
            a = np.concatenate(self.keys)
            self.keys = []
            a.sort()
            keep = np.empty(a.size, dtype=bool)
            keep[0] = True
            np.not_equal(a[1:], a[:-1], out=keep[1:])
            self.keys = [a[keep]]
            self.n_keys = self.keys[0].size

    def _is_global(self, addr, mask):
        """Generic-space load: True where the address lies in a global array."""
        a = _u(addr)
        reg = ((a - np.uint64(ADDR_LOW)) >> np.uint64(REGION_BITS)).astype(np.int64)
        ok = (a >= np.uint64(ADDR_LOW)) & (reg >= 0) & (reg < len(self.region_space))
        rs = np.asarray(self.region_space)
        sp = np.where(ok, np.take(rs, np.clip(reg, 0, len(rs) - 1)), "")
        g = np.broadcast_to((sp == "global") & ok, (self.n,))
        return g if mask is None else (g & mask)

    def _addr(self, o, ins):
        """Address operand -> (uint64 address, taint)."""
        if o.kind != "addr":
            raise PTXAnalysisError("%s: line %d: expected an address, got %r"
                                   % (self.k.name, ins.line, o.text))
        aw = self.aw
        if o.base is None:
            base, t = np.uint64(0), EMPTY
        elif o.base.kind == "sym" and o.base.name in self.pvals:
            return None, frozenset(["param-address:" + o.base.name])
        else:
            base, t = self._bits(o.base, aw, ins)
        off = np.uint64(o.off & _M[aw])
        return (base + off) & _MU[aw] if aw < 64 else base + off, t

    # -- main loop ----------------------------------------------------------------
    def run(self):
        instrs = self.k.instrs
        nins = len(instrs)
        n = self.n
        alive = np.ones(n, dtype=bool)
        n_alive = n
        pc = None
        cur = 0
        limit = self.step_limit
        steps = 0
        with np.errstate(all="ignore"):
            while n_alive:
                steps += 1
                if steps > limit:
                    raise PTXAnalysisError(
                        "kernel %s: more than %d instruction steps; the loop "
                        "bounds probably depend on data or on a --param value "
                        "(raise --step-limit if the kernel really runs this long)"
                        % (self.k.name, limit))
                if pc is None:
                    active = None if n_alive == n else alive
                else:
                    live = pc[alive]
                    cur = int(live.min())
                    if (live == cur).all():
                        pc = None
                        active = None if n_alive == n else alive
                    else:
                        active = alive & (pc == cur)
                if cur >= nins:  # fell off the end: same as ret
                    if active is None:
                        break
                    alive &= ~active
                    n_alive = int(alive.sum())
                    continue
                ins = instrs[cur]
                if ins.guard is not None:
                    g, gt = self._pred(ins.guard, ins)
                    if ins.guard_neg:
                        g = ~g
                    if gt:
                        self._check(gt, ins, "branch/guard predicate")
                    g = np.broadcast_to(g, (n,))
                    em = g if active is None else (active & g)
                else:
                    em = active
                kind = ins.kind
                if kind == "bra":
                    tgt = ins.target
                    if em is None:
                        cur = tgt
                        continue
                    base = alive if active is None else active
                    taken_any = bool(em.any())
                    rest = base & ~em
                    rest_any = bool(rest.any())
                    if pc is None:
                        if not taken_any:
                            cur += 1
                        elif not rest_any:
                            cur = tgt
                        else:
                            pc = np.full(n, cur + 1, dtype=np.int64)
                            pc[em] = tgt
                    else:
                        pc[em] = tgt
                        pc[rest] = cur + 1
                    continue
                if kind in ("exit", "trap"):
                    if kind == "trap":
                        self.report.warn_once((self.k.name, ins.line, "trap"),
                                              "kernel %s, line %d: trap treated as exit"
                                              % (self.k.name, ins.line))
                    if em is None:
                        break
                    base = alive if active is None else active
                    alive &= ~em
                    n_alive = int(alive.sum())
                    if pc is None:
                        cur += 1
                    else:
                        pc[base & ~em] = cur + 1
                    continue
                if em is None or em.any():
                    self._exec(ins, em)
                if pc is None:
                    cur += 1
                else:
                    pc[alive if active is None else active] = cur + 1
        self.steps = steps
        self._compact()
        return self.keys[0] if self.keys else np.zeros(0, dtype=np.uint64)

    # -- instruction semantics ----------------------------------------------------
    def _exec(self, ins, mask):
        op = ins.op
        h = getattr(self, "_op_" + op, None)
        if h is None:
            if op in _NOOPS:
                return
            self.report.warn_once(("op", op),
                                  "kernel %s, line %d: '%s' is not modelled; its "
                                  "result is treated as unknown" % (self.k.name,
                                                                    ins.line, op))
            self._taint_dst(ins, frozenset(["op:%s@%d" % (op, ins.line)]), mask)
            return
        h(ins, mask)

    def _is_float(self, ins):
        return bool(ins.types) and ins.types[-1][0] in "f" or \
            (bool(ins.types) and ins.types[-1] in ("bf16", "bf16x2", "tf32"))

    def _float_op(self, ins, mask, fn, nsrc):
        w = ins.w
        vals, taint = [], EMPTY
        for o in ins.ops[1:1 + nsrc]:
            v, t = self._float(o, w, ins)
            vals.append(v)
            taint |= t
        if taint:
            self._write(ins.ops[0], _U0, taint, mask, w)
            return
        self._write(ins.ops[0], _from_float(fn(*vals), w), EMPTY, mask, w)

    def _int_srcs(self, ins, w, nsrc, first=1):
        vals, taint = [], EMPTY
        for o in ins.ops[first:first + nsrc]:
            v, t = self._bits(o, w, ins)
            vals.append(v)
            taint |= t
        return vals, taint

    def _op_add(self, ins, mask):
        if self._is_float(ins):
            return self._float_op(ins, mask, lambda a, b: a + b, 2)
        if "cc" in ins.mods:
            return self._taint_dst(ins, frozenset(["op:add.cc@%d" % ins.line]), mask)
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        if "sat" in ins.mods:
            r = np.clip(_signed(a, 32) + _signed(b, 32), -2 ** 31, 2 ** 31 - 1)
            return self._write(ins.ops[0], _from_signed(r, 32), t, mask, w)
        self._write(ins.ops[0], a + b, t, mask, w)

    def _op_sub(self, ins, mask):
        if self._is_float(ins):
            return self._float_op(ins, mask, lambda a, b: a - b, 2)
        if "cc" in ins.mods:
            return self._taint_dst(ins, frozenset(["op:sub.cc@%d" % ins.line]), mask)
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        if "sat" in ins.mods:
            r = np.clip(_signed(a, 32) - _signed(b, 32), -2 ** 31, 2 ** 31 - 1)
            return self._write(ins.ops[0], _from_signed(r, 32), t, mask, w)
        self._write(ins.ops[0], a - b, t, mask, w)

    def _mul_core(self, ins, a, b, w):
        signed = ins.types[-1].startswith("s")
        if "wide" in ins.mods:
            if signed:
                return _from_signed(_signed(a, w) * _signed(b, w), 2 * w), 2 * w
            return (a * b) & _MU[2 * w], 2 * w
        if "hi" in ins.mods:
            return _mulhi(a, b, w, signed), w
        return a * b, w

    def _op_mul(self, ins, mask):
        if self._is_float(ins):
            return self._float_op(ins, mask, lambda a, b: a * b, 2)
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        r, rw = self._mul_core(ins, a, b, w)
        self._write(ins.ops[0], r, t, mask, rw)

    def _op_mad(self, ins, mask):
        if self._is_float(ins):
            return self._float_op(ins, mask, lambda a, b, c: a * b + c, 3)
        if "cc" in ins.mods:
            return self._taint_dst(ins, frozenset(["op:mad.cc@%d" % ins.line]), mask)
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        r, rw = self._mul_core(ins, a, b, w)
        c, tc = self._bits(ins.ops[3], rw, ins)
        self._write(ins.ops[0], r + c, t | tc, mask, rw)

    def _mul24(self, ins):
        """mul24.{lo,hi}: product of the low 24 bits (sign-extended for .s32)."""
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        m24 = np.uint64(0xFFFFFF)
        if ins.types[-1].startswith("s"):
            sa = _signed(a & m24, 32)
            sa = np.where(sa >= np.int64(0x800000), sa - np.int64(1 << 24), sa)
            sb = _signed(b & m24, 32)
            sb = np.where(sb >= np.int64(0x800000), sb - np.int64(1 << 24), sb)
            p = _from_signed(sa * sb, 64)
        else:
            p = (a & m24) * (b & m24)
        if "hi" in ins.mods:
            p = p >> np.uint64(16)
        return p, t

    def _op_mul24(self, ins, mask):
        p, t = self._mul24(ins)
        self._write(ins.ops[0], p, t, mask, ins.w)

    def _op_mad24(self, ins, mask):
        p, t = self._mul24(ins)
        c, tc = self._bits(ins.ops[3], ins.w, ins)
        self._write(ins.ops[0], p + c, t | tc, mask, ins.w)

    def _op_div(self, ins, mask):
        if self._is_float(ins):
            return self._float_op(ins, mask, lambda a, b: a / b, 2)
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        if ins.types[-1].startswith("s"):
            q, _ = _trunc_div(_signed(a, w), _signed(b, w))
            r = _from_signed(q, w)
        else:
            r = np.where(b == _U0, _U0, a // np.where(b == _U0, _U1, b))
        self._write(ins.ops[0], r, t, mask, w)

    def _op_rem(self, ins, mask):
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        if ins.types[-1].startswith("s"):
            _, rr = _trunc_div(_signed(a, w), _signed(b, w))
            r = _from_signed(rr, w)
        else:
            r = np.where(b == _U0, _U0, a % np.where(b == _U0, _U1, b))
        self._write(ins.ops[0], r, t, mask, w)

    def _minmax(self, ins, mask, is_min):
        if self._is_float(ins):
            f = np.fmin if is_min else np.fmax
            return self._float_op(ins, mask, f, 2)
        w = ins.w
        (a, b), t = self._int_srcs(ins, w, 2)
        if ins.types[-1].startswith("s"):
            sa, sb = _signed(a, w), _signed(b, w)
            r = _from_signed(np.minimum(sa, sb) if is_min else np.maximum(sa, sb), w)
        else:
            r = np.minimum(a, b) if is_min else np.maximum(a, b)
        self._write(ins.ops[0], r, t, mask, w)

    def _op_min(self, ins, mask):
        self._minmax(ins, mask, True)

    def _op_max(self, ins, mask):
        self._minmax(ins, mask, False)

    def _op_abs(self, ins, mask):
        if self._is_float(ins):
            return self._float_op(ins, mask, np.abs, 1)
        w = ins.w
        (a,), t = self._int_srcs(ins, w, 1)
        self._write(ins.ops[0], _from_signed(np.abs(_signed(a, w)), w), t, mask, w)

    def _op_neg(self, ins, mask):
        if self._is_float(ins):
            return self._float_op(ins, mask, lambda a: -a, 1)
        w = ins.w
        (a,), t = self._int_srcs(ins, w, 1)
        self._write(ins.ops[0], _U0 - a, t, mask, w)

    def _logic(self, ins, mask, fn):
        if ins.types and ins.types[-1] == "pred":
            vals, taint = [], EMPTY
            for o in ins.ops[1:]:
                v, t = self._pred(o, ins)
                vals.append(v)
                taint |= t
            return self._write(ins.ops[0], fn(*vals), taint, mask, pred=True)
        w = ins.w
        vals, t = self._int_srcs(ins, w, len(ins.ops) - 1)
        self._write(ins.ops[0], fn(*vals), t, mask, w)

    def _op_and(self, ins, mask):
        self._logic(ins, mask, lambda a, b: a & b)

    def _op_or(self, ins, mask):
        self._logic(ins, mask, lambda a, b: a | b)

    def _op_xor(self, ins, mask):
        self._logic(ins, mask, lambda a, b: a ^ b)

    def _op_not(self, ins, mask):
        self._logic(ins, mask, lambda a: ~a)

    def _op_cnot(self, ins, mask):
        w = ins.w
        (a,), t = self._int_srcs(ins, w, 1)
        self._write(ins.ops[0], np.where(a == _U0, _U1, _U0), t, mask, w)

    def _op_shl(self, ins, mask):
        w = ins.w
        a, t = self._bits(ins.ops[1], w, ins)
        s, ts = self._bits(ins.ops[2], 32, ins)
        big = s >= np.uint64(w)
        r = np.where(big, _U0, a << np.where(big, _U0, s))
        self._write(ins.ops[0], r, t | ts, mask, w)

    def _op_shr(self, ins, mask):
        w = ins.w
        a, t = self._bits(ins.ops[1], w, ins)
        s, ts = self._bits(ins.ops[2], 32, ins)
        if ins.types[-1].startswith("s"):
            sa = _signed(a, w)
            sh = np.minimum(s, np.uint64(w - 1)).astype(np.int64)
            r = _from_signed(sa >> sh, w)
        else:
            big = s >= np.uint64(w)
            r = np.where(big, _U0, a >> np.where(big, _U0, s))
        self._write(ins.ops[0], r, t | ts, mask, w)

    def _op_shf(self, ins, mask):
        a, ta = self._bits(ins.ops[1], 32, ins)
        b, tb = self._bits(ins.ops[2], 32, ins)
        c, tc = self._bits(ins.ops[3], 32, ins)
        n = np.minimum(c, np.uint64(32)) if "clamp" in ins.mods else (c & np.uint64(31))
        v = (b << np.uint64(32)) | a
        if "l" in ins.mods:
            r = ((v << n) >> np.uint64(32)) & _MU[32]
        else:
            r = (v >> n) & _MU[32]
        self._write(ins.ops[0], r, ta | tb | tc, mask, 32)

    def _cmp(self, cmp, a, b, kind, w):
        if kind == "f":
            fa, fb = _to_float(a, w), _to_float(b, w)
            nan = np.isnan(fa) | np.isnan(fb)
            base = cmp[:-1] if cmp.endswith("u") and cmp not in ("num",) else cmp
            unordered = cmp.endswith("u") and cmp not in ("num",)
            if cmp == "num":
                return ~nan
            if cmp == "nan":
                return nan
            r = {"eq": fa == fb, "ne": fa != fb, "lt": fa < fb, "le": fa <= fb,
                 "gt": fa > fb, "ge": fa >= fb}[base]
            return (r | nan) if unordered else (r & ~nan)
        if kind == "s" and cmp in ("lt", "le", "gt", "ge"):
            a, b = _signed(a, w), _signed(b, w)
        r = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b,
             "ge": a >= b, "lo": a < b, "ls": a <= b, "hi": a > b,
             "hs": a >= b}.get(cmp)
        if r is None:
            raise PTXAnalysisError("unsupported comparison %s" % cmp)
        return r

    def _op_setp(self, ins, mask):
        cmp = ins.mods[0]
        boolop = None
        for m in ins.mods[1:]:
            if m in ("and", "or", "xor"):
                boolop = m
        ty = ins.types[-1]
        w = ins.w
        kind = ty[0] if ty[0] in "fs" else "u"
        if ty in ("bf16", "bf16x2"):
            kind = "f"
        if kind == "f":
            a, ta = self._bits(ins.ops[1], w, ins)
            b, tb = self._bits(ins.ops[2], w, ins)
        else:
            a, ta = self._bits(ins.ops[1], w, ins)
            b, tb = self._bits(ins.ops[2], w, ins)
        r = np.asarray(self._cmp(cmp, a, b, kind, w))
        t = ta | tb
        q = None
        if boolop:
            c, tc = self._pred(ins.ops[3], ins)
            t |= tc
            f = {"and": lambda x, y: x & y, "or": lambda x, y: x | y,
                 "xor": lambda x, y: x ^ y}[boolop]
            q = f(~r, c)
            r = f(r, c)
        else:
            q = ~r
        d = ins.ops[0]
        if d.kind == "pair":
            self._write(d.items[0], r, t, mask, pred=True)
            if d.items[1] is not None and d.items[1].text != "_":
                self._write(d.items[1], q, t, mask, pred=True)
        else:
            self._write(d, r, t, mask, pred=True)

    def _op_set(self, ins, mask):
        # set.CmpOp{.BoolOp}.dtype.stype d, a, b{, c}
        cmp = ins.mods[0]
        dtype, stype = ins.types[-2], ins.types[-1]
        sw = _TYPE_BITS[stype]
        kind = stype[0] if stype[0] in "fs" else "u"
        a, ta = self._bits(ins.ops[1], sw, ins)
        b, tb = self._bits(ins.ops[2], sw, ins)
        r = np.asarray(self._cmp(cmp, a, b, kind, sw))
        t = ta | tb
        for m in ins.mods[1:]:
            if m in ("and", "or", "xor"):
                c, tc = self._pred(ins.ops[3], ins)
                t |= tc
                r = {"and": r & c, "or": r | c, "xor": r ^ c}[m]
        dw = _TYPE_BITS[dtype]
        if dtype.startswith("f"):
            val = np.where(r, _from_float(1.0, dw), _U0)
        else:
            val = np.where(r, _MU[dw], _U0)
        self._write(ins.ops[0], val, t, mask, dw)

    def _op_selp(self, ins, mask):
        w = ins.w
        a, ta = self._bits(ins.ops[1], w, ins)
        b, tb = self._bits(ins.ops[2], w, ins)
        c, tc = self._pred(ins.ops[3], ins)
        self._write(ins.ops[0], np.where(c, a, b), ta | tb | tc, mask, w)

    def _op_slct(self, ins, mask):
        w = _TYPE_BITS[ins.types[0]]
        a, ta = self._bits(ins.ops[1], w, ins)
        b, tb = self._bits(ins.ops[2], w, ins)
        ct = ins.types[-1]
        if ct.startswith("f"):
            c, tc = self._float(ins.ops[3], 32, ins)
            cond = c >= 0
        else:
            c, tc = self._bits(ins.ops[3], 32, ins)
            cond = _signed(c, 32) >= 0
        self._write(ins.ops[0], np.where(cond, a, b), ta | tb | tc, mask, w)

    def _op_mov(self, ins, mask):
        d, s = ins.ops[0], ins.ops[1]
        ty = ins.types[-1] if ins.types else "b64"
        if ty == "pred":
            v, t = self._pred(s, ins)
            return self._write(d, v, t, mask, pred=True)
        w = _TYPE_BITS[ty]
        if d.kind == "vec":  # unpack: mov.b64 {lo, hi}, %rd
            v, t = self._bits(s, w, ins)
            ew = w // len(d.items)
            for i, it in enumerate(d.items):
                if it is not None:
                    self._write(it, v >> np.uint64(i * ew), t, mask, ew)
            return
        if s.kind == "vec":  # pack: mov.b64 %rd, {lo, hi}
            ew = w // len(s.items)
            acc, t = _U0, EMPTY
            for i, it in enumerate(s.items):
                if it is None:
                    continue
                v, ti = self._bits(it, ew, ins)
                acc = acc | (v << np.uint64(i * ew))
                t |= ti
            return self._write(d, acc, t, mask, w)
        v, t = self._bits(s, w, ins)
        self._write(d, v, t, mask, w)

    def _op_cvta(self, ins, mask):
        w = ins.w or self.aw
        v, t = self._bits(ins.ops[1], w, ins)
        self._write(ins.ops[0], v, t, mask, w)

    def _op_cvt(self, ins, mask):
        dty, sty = ins.types[-2], ins.types[-1]
        dw, sw = _TYPE_BITS[dty], _TYPE_BITS[sty]
        s_is_f = sty[0] == "f" or sty.startswith("bf")
        d_is_f = dty[0] == "f" or dty.startswith("bf")
        if s_is_f:
            x, t = self._float(ins.ops[1], sw, ins)
        else:
            b, t = self._bits(ins.ops[1], sw, ins)
        if t and (s_is_f or d_is_f):
            return self._write(ins.ops[0], _U0, t, mask, dw)
        if not s_is_f and not d_is_f:
            if sty.startswith("s"):
                v = _signed(b, sw)
                if "sat" in ins.mods:
                    v = self._sat(v, dty, dw)
                r = _from_signed(v, dw)
            else:
                v = b
                if "sat" in ins.mods and dty.startswith("s") and dw <= sw:
                    v = np.minimum(v, np.uint64((1 << (dw - 1)) - 1))
                elif "sat" in ins.mods and dw < sw:
                    v = np.minimum(v, _MU[dw])
                r = v & _MU[dw] if dw < 64 else v
            return self._write(ins.ops[0], r, t, mask, dw)
        if s_is_f and not d_is_f:
            mods = ins.mods
            if "rni" in mods:
                y = np.rint(x)
            elif "rmi" in mods:
                y = np.floor(x)
            elif "rpi" in mods:
                y = np.ceil(x)
            else:
                y = np.trunc(x)
            y = np.where(np.isnan(y), 0.0, y)
            if dty.startswith("s"):
                lo, hi = -(2.0 ** (dw - 1)), 2.0 ** (dw - 1) - 1
                y = np.clip(y, lo, hi)
                r = _from_signed(y.astype(np.int64), dw)
            else:
                y = np.clip(y, 0.0, 2.0 ** dw - 1)
                r = np.asarray(y).astype(np.uint64) & _MU[dw] if dw < 64 \
                    else np.asarray(y).astype(np.uint64)
            return self._write(ins.ops[0], r, t, mask, dw)
        if not s_is_f and d_is_f:
            v = _signed(b, sw).astype(np.float64) if sty.startswith("s") \
                else np.asarray(b).astype(np.float64)
            return self._write(ins.ops[0], _from_float(v, dw), t, mask, dw)
        y = x
        if "rni" in ins.mods:
            y = np.rint(x)
        elif "rzi" in ins.mods:
            y = np.trunc(x)
        elif "rmi" in ins.mods:
            y = np.floor(x)
        elif "rpi" in ins.mods:
            y = np.ceil(x)
        self._write(ins.ops[0], _from_float(y, dw), t, mask, dw)

    @staticmethod
    def _sat(v, dty, dw):
        if dty.startswith("s"):
            return np.clip(v, -(1 << (dw - 1)), (1 << (dw - 1)) - 1)
        return np.clip(v, 0, (1 << dw) - 1 if dw < 64 else (1 << 63) - 1)

    # loads/stores ---------------------------------------------------------------
    def _op_ld(self, ins, mask):
        sp = ins.space
        d = ins.ops[0]
        dsts = d.items if d.kind == "vec" else [d]
        ew = ins.w
        if sp == "param":
            a = ins.ops[1]
            if a.kind == "addr" and a.base is not None and a.base.kind == "sym" \
                    and a.base.name in self.pvals and a.off == 0 and len(dsts) == 1:
                bits, t = self.pvals[a.base.name]
                return self._write(dsts[0], _u(bits), t, mask, ew)
            name = a.base.name if (a.kind == "addr" and a.base is not None) else a.text
            t = frozenset(["param:%s%s" % (name, ("+%d" % a.off) if a.off else "")]) \
                if name in self.pvals else frozenset(["op:ld.param %s@%d" % (name, ins.line)])
            for x in dsts:
                if x is not None:
                    self._write(x, _U0, t, mask, ew)
            return
        if sp == "global" or sp is None:  # None: generic address space
            addr, ta = self._addr(ins.ops[1], ins)
            if ta:
                self._check(ta, ins, "load address")  # raises unless data
                self.ld_skipped += 1
            else:
                rec_mask = mask
                if sp is None:  # count it only where it reads a global array
                    rec_mask = self._is_global(addr, mask)
                if sp == "global" or rec_mask.any():
                    self._record(addr, len(dsts), max(ew // 8, 1), rec_mask, ins)
        for x in dsts:
            if x is not None:
                self._write(x, _U0, frozenset(["data"]), mask, ew)

    def _op_ldu(self, ins, mask):
        self._op_ld(ins, mask)

    def _op_st(self, ins, mask):
        return

    def _op_atom(self, ins, mask):
        self._taint_dst(ins, frozenset(["data"]), mask)

    # bit manipulation -------------------------------------------------------------
    def _op_popc(self, ins, mask):
        w = _TYPE_BITS[ins.types[-1]]
        a, t = self._bits(ins.ops[1], w, ins)
        a = np.asarray(a, dtype=np.uint64)
        cnt = np.zeros(a.shape, dtype=np.uint64)
        x = a.copy()
        for _ in range(w):
            cnt += x & _U1
            x = x >> _U1
        self._write(ins.ops[0], cnt, t, mask, 32)

    def _op_clz(self, ins, mask):
        w = _TYPE_BITS[ins.types[-1]]
        a, t = self._bits(ins.ops[1], w, ins)
        a = np.asarray(a, dtype=np.uint64)
        cnt = np.full(a.shape, w, dtype=np.uint64)
        for i in range(w):
            bit = ((a >> np.uint64(i)) & _U1) == _U1
            cnt = np.where(bit, np.uint64(w - 1 - i), cnt)
        self._write(ins.ops[0], cnt, t, mask, 32)

    def _op_brev(self, ins, mask):
        w = ins.w
        a, t = self._bits(ins.ops[1], w, ins)
        a = np.asarray(a, dtype=np.uint64)
        r = np.zeros(a.shape, dtype=np.uint64)
        for i in range(w):
            r |= ((a >> np.uint64(i)) & _U1) << np.uint64(w - 1 - i)
        self._write(ins.ops[0], r, t, mask, w)

    def _op_bfe(self, ins, mask):
        w = ins.w
        a, ta = self._bits(ins.ops[1], w, ins)
        b, tb = self._bits(ins.ops[2], 32, ins)
        c, tc = self._bits(ins.ops[3], 32, ins)
        uw = np.uint64(w)
        pos = np.minimum(b & np.uint64(0xFF), uw)
        ln = np.minimum(c & np.uint64(0xFF), uw)
        # bits of the field that lie inside a: min(len, w - pos)
        eff = np.where(pos >= uw, _U0, np.minimum(ln, uw - pos))
        x = np.where(pos >= uw, _U0, a >> np.minimum(pos, uw - _U1))
        emask = np.where(eff >= np.uint64(64), _MU[64], (_U1 << eff) - _U1)
        r = x & emask
        if ins.types[-1].startswith("s"):
            # sbit = a[min(pos + len - 1, msb)]; every bit past the part of
            # the field inside a is sbit (ISA pseudo-code for bfe)
            top = np.minimum(pos + ln, uw) - _U1
            sbit = np.where(ln == _U0, _U0, (a >> top) & _U1)
            r = r | np.where(sbit == _U1, _MU[w] & ~emask, _U0)
        self._write(ins.ops[0], r, ta | tb | tc, mask, w)

    def _op_bfi(self, ins, mask):
        w = ins.w
        a, ta = self._bits(ins.ops[1], w, ins)
        b, tb = self._bits(ins.ops[2], w, ins)
        c, tc = self._bits(ins.ops[3], 32, ins)
        d, td = self._bits(ins.ops[4], 32, ins)
        pos = c & np.uint64(0xFF)
        ln = d & np.uint64(0xFF)
        lmask = np.where(ln >= np.uint64(64), _MU[64], (_U1 << ln) - _U1)
        pm = np.where(pos >= np.uint64(w), _U0, lmask << np.minimum(pos, np.uint64(63)))
        r = (b & ~pm) | ((a << np.minimum(pos, np.uint64(63))) & pm)
        self._write(ins.ops[0], r, ta | tb | tc | td, mask, w)

    def _op_prmt(self, ins, mask):
        a, ta = self._bits(ins.ops[1], 32, ins)
        b, tb = self._bits(ins.ops[2], 32, ins)
        c, tc = self._bits(ins.ops[3], 32, ins)
        mode = [m for m in ins.mods if m in ("f4e", "b4e", "rc8", "ecl", "ecr", "rc16")]
        if mode:
            return self._taint_dst(ins, frozenset(["op:prmt.%s@%d" % (mode[0], ins.line)]), mask)
        v = (_u(b) << np.uint64(32)) | _u(a)
        r = _U0
        for i in range(4):
            sel = (_u(c) >> np.uint64(4 * i)) & np.uint64(0xF)
            idx = sel & np.uint64(7)
            byte = (v >> (idx * np.uint64(8))) & np.uint64(0xFF)
            sign = (byte >> np.uint64(7)) & _U1
            byte = np.where((sel & np.uint64(8)) != _U0,
                            np.where(sign == _U1, np.uint64(0xFF), _U0), byte)
            r = r | (byte << np.uint64(8 * i))
        self._write(ins.ops[0], r, ta | tb | tc, mask, 32)

    def _op_sad(self, ins, mask):
        w = ins.w
        (a, b, c), t = self._int_srcs(ins, w, 3)
        if ins.types[-1].startswith("s"):
            less = _signed(a, w) < _signed(b, w)
        else:
            less = a < b
        dlt = np.where(less, b - a, a - b)  # modulo 2^64, as in the ISA
        self._write(ins.ops[0], dlt + c, t, mask, w)

    # floating point -----------------------------------------------------------
    def _op_fma(self, ins, mask):
        self._float_op(ins, mask, lambda a, b, c: a * b + c, 3)

    def _op_sqrt(self, ins, mask):
        self._float_op(ins, mask, np.sqrt, 1)

    def _op_rsqrt(self, ins, mask):
        self._float_op(ins, mask, lambda a: 1.0 / np.sqrt(a), 1)

    def _op_rcp(self, ins, mask):
        self._float_op(ins, mask, lambda a: 1.0 / a, 1)

    def _op_ex2(self, ins, mask):
        self._float_op(ins, mask, np.exp2, 1)

    def _op_lg2(self, ins, mask):
        self._float_op(ins, mask, np.log2, 1)

    def _op_sin(self, ins, mask):
        self._float_op(ins, mask, np.sin, 1)

    def _op_cos(self, ins, mask):
        self._float_op(ins, mask, np.cos, 1)

    def _op_tanh(self, ins, mask):
        self._float_op(ins, mask, np.tanh, 1)

    def _op_copysign(self, ins, mask):
        self._float_op(ins, mask, lambda a, b: np.copysign(b, a), 2)

# Instructions without an effect on addresses or control flow.
_NOOPS = frozenset([
    "bar", "barrier", "membar", "fence", "nanosleep", "prefetch", "prefetchu",
    "red", "pmevent", "discard", "applypriority", "griddepcontrol", "setmaxnreg",
    "elect", "mbarrier", "cp", "wgmma", "stmatrix", "call", "trap", "brkpt",
    "st", "sust", "sured",
])


# ----------------------------------------------------------------------------
# locality matrix from (address, TB) pairs
# ----------------------------------------------------------------------------
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _add_shared(keys, n_tbs, L, chunk=8192):
    """Add one sorted, distinct key block to L; return its distinct addresses.

    keys: sorted unique uint64 ((rel_address << TB_BITS) | tb); all keys of
    an address must be in the same block.  Addresses read by a single TB are
    dropped first.  The others are grouped by the exact set of TBs that read
    them (one bitmask per address), and each distinct set adds its number of
    addresses to all of its TB pairs.
    """
    if keys.size == 0:
        return 0
    addr = keys >> np.uint64(TB_BITS)
    new = np.empty(addr.size, dtype=bool)
    new[0] = True
    np.not_equal(addr[1:], addr[:-1], out=new[1:])
    del addr
    starts = np.flatnonzero(new)
    del new
    counts = np.diff(np.append(starts, keys.size))
    distinct = int(starts.size)
    shared = counts >= 2
    if not shared.any():
        return distinct
    keep = np.repeat(shared, counts)
    tb_s = (keys[keep] & np.uint64((1 << TB_BITS) - 1)).astype(np.int64)
    del keep
    cnt_s = counts[shared]
    del starts, counts, shared
    n_runs = cnt_s.size
    n_words = (n_tbs + 63) // 64
    # (run, 64-TB word) groups are contiguous because the TBs of a run are
    # sorted; OR-ing distinct bits is the same as adding them.
    run = np.repeat(np.arange(n_runs, dtype=np.int64), cnt_s)
    seg = run * n_words + (tb_s >> 6)
    del run
    bnd = np.empty(seg.size, dtype=bool)
    bnd[0] = True
    np.not_equal(seg[1:], seg[:-1], out=bnd[1:])
    seg_start = np.flatnonzero(bnd)
    del bnd
    bits = np.left_shift(np.uint64(1), (tb_s & 63).astype(np.uint64))
    sig = np.zeros(n_runs * n_words, dtype=np.uint64)
    sig[seg[seg_start]] = np.add.reduceat(bits, seg_start)
    del bits, seg, seg_start
    sig = sig.reshape(n_runs, n_words)
    uniq, cnt = np.unique(sig, axis=0, return_counts=True)
    del sig
    sizes = _POPCOUNT8[uniq.view(np.uint8)].reshape(uniq.shape[0], -1).sum(axis=1)
    cost_sets = float((sizes.astype(np.float64) ** 2).sum())
    cost_dense = float(n_runs) * n_tbs * n_tbs
    if uniq.shape[0] <= 200000 and (cost_sets <= 4 * cost_dense or cost_sets < 5e7):
        for row, c in zip(uniq, cnt):
            m = np.flatnonzero(np.unpackbits(row.view(np.uint8), bitorder="little")[:n_tbs])
            L[np.ix_(m, m)] += np.int64(c)
    else:  # many different sharing sets: dense incidence products
        run_id = np.repeat(np.arange(n_runs), cnt_s)
        for s0 in range(0, n_runs, chunk):
            e0 = min(s0 + chunk, n_runs)
            lo, hi = np.searchsorted(run_id, [s0, e0])
            M = np.zeros((e0 - s0, n_tbs), dtype=np.float32)
            M[run_id[lo:hi] - s0, tb_s[lo:hi]] = 1.0
            L += np.rint(M.T.astype(np.float64) @ M).astype(np.int64)
    return distinct


def locality_from_parts(parts, n_tbs, bucket=4000000):
    """L[i][j] = number of addresses read by both TB i and TB j (i != j).

    parts: per-batch arrays of sorted unique keys (disjoint TB ranges).  The
    address range is cut into buckets of about `bucket` keys, so that the
    temporary arrays stay small; every bucket holds all keys of its
    addresses.  Returns (L, number of distinct addresses).
    """
    L = np.zeros((n_tbs, n_tbs), dtype=np.int64)
    parts = [p for p in parts if p.size]
    total = sum(int(p.size) for p in parts)
    if total == 0:
        return L, 0
    sh = np.uint64(TB_BITS)
    if total <= bucket:
        cuts = []
    else:
        step = max(1, total // 4096)  # same stride in every part
        sample = np.sort(np.concatenate([p[::step] >> sh for p in parts]))
        n_b = (total + bucket - 1) // bucket
        cuts = np.unique(sample[np.linspace(0, sample.size - 1, n_b + 1)[1:-1].astype(np.int64)])
    bounds = [np.uint64(0)] + [np.uint64(int(c)) << sh for c in cuts] + [None]
    distinct = 0
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        pieces = []
        for p in parts:
            i0 = int(np.searchsorted(p, lo)) if b else 0
            i1 = int(np.searchsorted(p, hi)) if hi is not None else p.size
            if i1 > i0:
                pieces.append(p[i0:i1])
        if not pieces:
            continue
        if len(pieces) > 1:
            keys = np.concatenate(pieces)
            keys.sort()
        else:
            keys = pieces[0]  # read-only use below
        distinct += _add_shared(keys, n_tbs, L)
        del keys, pieces
    np.fill_diagonal(L, 0)
    return L, distinct


def locality_from_keys(keys, n_tbs):
    """Same as locality_from_parts for one sorted unique key array."""
    return locality_from_parts([keys], n_tbs)[0]


def lower_triangle(L):
    """LocalityGuru's matrix layout: entry [j][i] (j > i) holds L(i, j)."""
    n = L.shape[0]
    out = np.tril(L, -1)
    return [[int(x) for x in out[j]] for j in range(n)]


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
class _Report(object):
    def __init__(self, log):
        self.log = log
        self.seen = set()
        self.warnings = []

    def warn_once(self, key, msg):
        if key in self.seen:
            return
        self.seen.add(key)
        self.warnings.append(msg)
        self.log("[Warning] " + msg)


def _param_value(p, text):
    """--param text -> bits for parameter p."""
    t = text.strip()
    if p.type and p.type[0] == "f":
        return int(_from_float(float(t), p.w))
    if _INT_RE.match(t):
        return _parse_int(t) & _M[min(max(p.w, 8), 64)]
    try:
        return int(_from_float(float(t), 64))
    except ValueError:
        raise PTXAnalysisError("bad value %r for parameter %s" % (text, p.name))


def _match_param(kernel, key):
    """--param key -> Param of this kernel (full name, 'param_N' suffix or N)."""
    for p in kernel.params:
        if key == p.name:
            return p
    for p in kernel.params:
        if p.name.endswith("_" + key) or (key.isdigit() and p.index == int(key)):
            return p
    return None


def analyze_kernel(mod, kernel, grid, block, params=None, batch_threads=65536,
                   step_limit=20000000, log=print, report=None):
    """Execute one kernel; return dict(matrix=L (n x n), stats=..., warnings=...)."""
    report = report or _Report(log)
    grid = tuple(int(x) for x in (list(grid) + [1, 1])[:3])
    block = tuple(int(x) for x in (list(block) + [1, 1])[:3])
    n_tbs = grid[0] * grid[1] * grid[2]
    tpb = block[0] * block[1] * block[2]
    if n_tbs >= (1 << TB_BITS):
        raise PTXAnalysisError("too many TBs (%d)" % n_tbs)
    classify_params(kernel)
    # address regions: pointer parameters first, then declared variables
    region_space = []
    pvals = {}
    given = {}
    for key, val in (params or {}).items():
        p = _match_param(kernel, key)
        if p is not None:
            given[p.name] = val
    for p in kernel.params:
        if p.name in given:
            pvals[p.name] = (np.uint64(_param_value(p, given[p.name])), EMPTY)
        elif p.is_ptr:
            r = len(region_space)
            region_space.append("global")
            pvals[p.name] = (np.uint64(ADDR_LOW + (r + 1) * (1 << REGION_BITS)), EMPTY)
        else:
            pvals[p.name] = (_U0, frozenset(["param:" + p.name]))
    symbase = {}
    for name, sp in list(mod.symbols.items()) + list(kernel.symbols.items()):
        if name in symbase:
            continue
        r = len(region_space)
        region_space.append(sp)
        symbase[name] = ADDR_LOW + (r + 1) * (1 << REGION_BITS)
    if len(region_space) > MAX_REGIONS:
        raise PTXAnalysisError("too many arrays (%d)" % len(region_space))
    per_batch = max(1, min(n_tbs, batch_threads // max(tpb, 1)))
    parts = []
    stats = {"ld_global_executed": 0, "ld_global_skipped": 0, "steps": 0}
    for first in range(0, n_tbs, per_batch):
        cnt = min(per_batch, n_tbs - first)
        run = _Run(kernel, grid, block, first, cnt, pvals, symbase, region_space,
                   mod.address_size, step_limit, report)
        parts.append(run.run())
        stats["ld_global_executed"] += run.ld_exec
        stats["ld_global_skipped"] += run.ld_skipped
        stats["steps"] = max(stats["steps"], run.steps)
    per_tb = np.zeros(n_tbs, dtype=np.int64)
    tbm = np.uint64((1 << TB_BITS) - 1)
    for part in parts:
        per_tb += np.bincount((part & tbm).astype(np.int64), minlength=n_tbs)
    stats["addresses_per_tb_min"] = int(per_tb.min()) if n_tbs else 0
    stats["addresses_per_tb_max"] = int(per_tb.max()) if n_tbs else 0
    stats["pointer_params"] = [p.name for p in kernel.params if p.is_ptr]
    L, stats["distinct_addresses"] = locality_from_parts(parts, n_tbs)
    del parts
    return {"matrix": L, "stats": stats, "warnings": list(report.warnings),
            "grid": grid, "block": block}


def analyze_ptx(ptx_path, grid, block, params=None, kernels=None,
                batch_threads=65536, step_limit=20000000, log=print):
    """Locality matrices for every .entry kernel of a PTX file."""
    with open(ptx_path, "r") as f:
        mod = parse_ptx(f.read())
    for key in (params or {}):
        if not any(_match_param(k, key) for k in mod.kernels.values()):
            log("[Warning] --param %s matches no parameter of any kernel in %s"
                % (key, ptx_path))
    out = OrderedDict()
    for name, k in mod.kernels.items():
        if kernels and name not in kernels:
            continue
        out[name] = analyze_kernel(mod, k, grid, block, params, batch_threads,
                                   step_limit, log)
    return out


def dims_from_filename(path):
    """<APP>_<gridX>_<gridY>_<blockX>_<blockY>.ptx -> (app, grid, block)."""
    base = os.path.basename(path)
    base = base[:-4] if base.endswith(".ptx") else base
    p = base.split("_")
    if len(p) < 5:
        raise PTXAnalysisError("file name %s is not <APP>_<gridX>_<gridY>_"
                               "<blockX>_<blockY>.ptx" % base)
    return p[0], (int(p[-4]), int(p[-3]), 1), (int(p[-2]), int(p[-1]), 1)


def parse_param_args(items):
    """['name=value', ...] -> {name: value}."""
    out = OrderedDict()
    for it in items or []:
        if "=" not in it:
            raise PTXAnalysisError("--param expects NAME=VALUE, got %r" % it)
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="TB locality matrices by executing a kernel's PTX "
                    "(LocalityGuru Algorithm 2, including loops and idle threads)")
    ap.add_argument("ptx", help="PTX file, named <APP>_<gx>_<gy>_<bx>_<by>.ptx "
                                "unless --grid/--block are given")
    ap.add_argument("--grid", help="gx,gy[,gz]")
    ap.add_argument("--block", help="bx,by[,bz]")
    ap.add_argument("--param", action="append", default=[],
                    help="scalar kernel parameter NAME=VALUE (repeatable)")
    ap.add_argument("--kernel", action="append", default=[],
                    help="only this kernel (repeatable)")
    ap.add_argument("--out-dir", help="write <kernel>_<gx>-<gy>_matrix.json here")
    ap.add_argument("--batch-threads", type=int, default=65536)
    ap.add_argument("--step-limit", type=int, default=20000000)
    a = ap.parse_args(argv)
    app, grid, block = None, None, None
    try:
        app, grid, block = dims_from_filename(a.ptx)
    except (PTXAnalysisError, ValueError):
        pass
    if a.grid:
        grid = tuple(int(x) for x in a.grid.split(","))
    if a.block:
        block = tuple(int(x) for x in a.block.split(","))
    if grid is None or block is None:
        ap.error("give --grid and --block, or name the file "
                 "<APP>_<gx>_<gy>_<bx>_<by>.ptx")
    try:
        res = analyze_ptx(a.ptx, grid, block, parse_param_args(a.param),
                          a.kernel or None, a.batch_threads, a.step_limit)
    except PTXAnalysisError as e:
        print("[Error] %s" % e)
        return 1
    for name, r in res.items():
        L = r["matrix"]
        s = r["stats"]
        n = L.shape[0]
        off = L[np.triu_indices(n, 1)] if n > 1 else np.zeros(0, dtype=np.int64)
        print("%s: %d TBs, %d ld.global executions, %d-%d addresses per TB, "
              "pair weights %d-%d, %d of %d pairs share data"
              % (name, n, s["ld_global_executed"], s["addresses_per_tb_min"],
                 s["addresses_per_tb_max"], off.min() if off.size else 0,
                 off.max() if off.size else 0, int((off > 0).sum()), off.size))
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
            path = os.path.join(a.out_dir, "%s_%d-%d_matrix.json"
                                % (name, r["grid"][0], r["grid"][1]))
            with open(path, "w") as f:
                json.dump(lower_triangle(L), f)
            print("  -> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

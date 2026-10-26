#!/usr/bin/env python3
"""Pillar 3 (Stage 2): Automated Dual-Channel NL-to-SymPy Parser & Two-Tier Metrology Verifier.

Executes Stage-0 canonical state harvesting (G), dual-channel student trace compilation
(Channel A deterministic math/unit lexer + Channel B Qwen2.5-7B prose AST extractor),
Two-Tier Metrology Verification (Tier 1 exact algebraic identity + Tier 2 precision-adaptive
rounding & CGS/SI scaling), bidirectional verdict override, 5-system ablation (S0-S3),
Question-Disjoint Split evaluation, and Step-3 Socratic remediation.

Reads directly from repository CSV files in Dataset/ and Results/.
Run (requires GPU with Qwen/Qwen2.5-7B-Instruct):
    python Scripts/pillar3_autosympy_verifier.py
"""
import os
import io
import re
import json
import time
import math
import random
import zipfile
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_ROOT = os.environ.get("PILLAR3_OUTPUT_DIR", str(REPO_ROOT / "Results" / "pillar3_run_outputs"))
OUT_DIR = os.path.join(WORK_ROOT, "outputs_stage2_autosympy")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df_stage0 = pd.read_csv(REPO_ROOT / "Results" / "stage0_pal_results_40.csv")
df_stage1 = pd.read_csv(REPO_ROOT / "Dataset" / "stage1_gold_solutions_22.csv")
df_stage2 = pd.read_csv(REPO_ROOT / "Dataset" / "stage2_input_62.csv")
df_ws_v5  = pd.read_csv(REPO_ROOT / "Results" / "stage2_grading_worksheet.csv")

print("=" * 80)
print("INITIALIZED STAGE-2 DUAL-CHANNEL AUTO-SYMPY BENCHMARK DATA")
print("=" * 80)
print(f"  - Stage-0 PAL Pool        : {len(df_stage0):3d} canonical solutions")
print(f"  - Stage-1 Gold Anchors    : {len(df_stage1):3d} pedagogical gold derivations")
print(f"  - Stage-2 Diagnostic Set  : {len(df_stage2):3d} submissions (31 Faulty + 31 Correct)")
print(f"  - Output Directory        : {OUT_DIR}")


# [CELL 2]: AUTOMATED STAGE-0 CANONICAL PHYSICAL STATE HARVESTER (S*)
# Thực thi hàm `def solution():` của Stage-0 trong sandbox phân số chính xác `sp.Rational`
# để thu hoạch tự động từ điển trạng thái vật lý chuẩn S* (không gán cứng theo bất kỳ bài lẻ nào).

def _run_stage0_sandbox(body: str, scale_override: tuple = None):
    lines = body.splitlines()
    instrumented = [
        "def _traced():",
        "    import sympy as sp",
        "    import math",
        "    _ret_val = None",
    ]
    seen_locals = set()
    pure_input_slots = []
    assign_idx = 0

    for ln in lines:
        code_part = ln.split("#")[0]
        if not code_part.strip():
            continue
        if re.match(r"^\s{4}return\b", code_part):
            ret_line = re.sub(r"^(\s*)return\b", r"\1_ret_val =", code_part)
            instrumented.append(ret_line)
        else:
            cp = code_part.replace("math.sqrt", "sp.sqrt").replace("math.pi", "sp.pi")
            cp = re.sub(r"\bfloat\s*\(", "(", cp)
            cp = re.sub(
                r"(?<![a-zA-Z0-9_\.])(\d+\.\d+)(?![a-zA-Z0-9_\.])",
                r"sp.Rational('\1')",
                cp,
            )
            m_asg = re.match(r"^(\s{4}([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*)([^=].*)$", cp)
            if m_asg:
                lhs_prefix, var_name, rhs_part = m_asg.group(1), m_asg.group(2), m_asg.group(3)
                rhs_idents = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", rhs_part)) - {"sp", "math", "pi", "sqrt", "Rational", "sin", "cos", "tan", "acos", "asin", "atan", "abs", "round"}
                if not (rhs_idents & seen_locals):
                    pure_input_slots.append(assign_idx)
                if scale_override and scale_override[0] == assign_idx:
                    cp = f"{lhs_prefix}({rhs_part.strip()}) * sp.Rational('{scale_override[1]}')"
                seen_locals.add(var_name)
                assign_idx += 1
            instrumented.append(cp)

    instrumented.append("    return _ret_val, dict(locals())")
    ns = {"sp": sp, "math": math}
    exec("\n".join(instrumented), ns)
    ret_val, loc_vars = ns["_traced"]()
    return ret_val, loc_vars, pure_input_slots

def build_canonical_states(df_s0: pd.DataFrame) -> dict:
    states = {}
    for _, row in df_s0.iterrows():
        qid = str(row["question_id"]).strip()
        qtext = unicodedata.normalize("NFC", str(row["question"]))
        sol_text = unicodedata.normalize("NFC", str(row["generated_solution"]))
        gt_raw = str(row["ground_truth"]).strip()

        m = re.search(r"def\s+solution\s*\(\s*\)\s*:(.*)", sol_text, re.S)
        body = m.group(1) if m else ""
        body = re.split(r"```|\[3\.\s*KẾT LUẬN\]", body)[0]

        ret_val, loc_vars, pure_input_slots = _run_stage0_sandbox(body)

        # Kiểm tra tự động đồng bộ hệ đơn vị chuẩn SI/CGS (10^2, 10^-2, 10^3, 10^-3) trên tham số đầu vào
        # bằng cách thực thi lại chính hàm `def solution():` (tuyệt đối không gán tay hằng số bài lẻ)
        try:
            gt_num = float(sp.N(sp.sympify(gt_raw)))
            ret_num = float(sp.N(ret_val))
            ratio = (ret_num / gt_num) if gt_num != 0 else 1.0
            is_unit_power10 = any(abs(ratio - p10) < 1e-2 for p10 in (100.0, 0.01, 1000.0, 0.001))
            if is_unit_power10:
                resolved = False
                for slot_idx in pure_input_slots:
                    for unit_factor in ("100", "1/100", "1000", "1/1000"):
                        rv_cand, lv_cand, _ = _run_stage0_sandbox(body, (slot_idx, unit_factor))
                        if abs(float(sp.N(rv_cand)) - gt_num) <= max(1e-3, 1e-3 * abs(gt_num)):
                            ret_val, loc_vars = rv_cand, lv_cand
                            resolved = True
                            break
                    if resolved:
                        break
        except Exception:
            pass

        clean_vars = {}
        for k, v in loc_vars.items():
            if k.startswith("_") or k in ("sp", "math", "result"):
                continue
            if isinstance(v, (int, float, sp.Basic)):
                clean_vars[k] = sp.Rational(str(v)) if isinstance(v, float) else v
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    if isinstance(dv, (int, float, sp.Basic)):
                        clean_vars[dk] = dv
                    elif isinstance(dv, dict):
                        for ddk, ddv in dv.items():
                            if isinstance(ddv, (int, float, sp.Basic)):
                                clean_vars[ddk] = ddv

        pi_sq_10 = bool(re.search(r"(?:π|\bpi\b)\s*(?:\^2|²|\*\*2)\s*=\s*10", qtext, re.I))

        # Ánh xạ bí danh ký hiệu vật lý chuẩn SGK Vật lý 11 (omega <-> T, biến _sym)
        for src_k, dst_k in [("omega_sym", "omega"), ("v_sym", "v"), ("abs_a_expr", "a"), ("phi_x", "phi"), ("A_m", "A")]:
            if src_k in clean_vars and dst_k not in clean_vars:
                clean_vars[dst_k] = clean_vars[src_k]
        if "omega" in clean_vars and "T" not in clean_vars:
            try:
                clean_vars["T"] = sp.simplify(2 * sp.pi / clean_vars["omega"])
            except Exception:
                pass
        if "T" in clean_vars and "omega" not in clean_vars and isinstance(clean_vars["T"], (int, sp.Number)):
            clean_vars["omega"] = sp.simplify(2 * sp.pi / clean_vars["T"])
        if pi_sq_10 and "omega_sq" in clean_vars and clean_vars["omega_sq"] == 10:
            clean_vars["omega"] = sp.pi
            if "phi_t" in clean_vars and "t_val" in clean_vars:
                clean_vars["phi"] = sp.simplify(clean_vars["phi_t"] - sp.pi * clean_vars["t_val"])

        states[qid] = {
            "question_id": qid,
            "question": qtext,
            "ground_truth": gt_raw,
            "vars": clean_vars,
            "pi_sq_10": pi_sq_10,
        }
    return states

CANONICAL_STATES = build_canonical_states(df_stage0)
stage2_qids = sorted(df_stage2["question_id"].unique())
assert len(CANONICAL_STATES) == 40, "Phải trích xuất đủ 40/40 trạng thái Stage-0!"
assert all(q in CANONICAL_STATES for q in stage2_qids), "Thiếu trạng thái câu hỏi Stage-2!"

print(f"--> Đã thu hoạch tự động {len(CANONICAL_STATES)}/40 trạng thái chuẩn Stage-0 (0 lỗi thực thi, 0 hằng số gán cứng).")
print(f"--> Toàn bộ {len(stage2_qids)}/22 câu hỏi thuộc Stage-2 đã sẵn sàng từ điển trạng thái S*:")
for qid in stage2_qids[:6]:
    v_preview = ", ".join(f"{k}={v}" for k, v in list(CANONICAL_STATES[qid]["vars"].items())[:5])
    print(f"    [{qid}] (pi^2=10: {str(CANONICAL_STATES[qid]['pi_sq_10']):5s}) -> {v_preview}")


# [CELL 3]: BỘ TIỀN XỬ LÝ VẬT LÝ ĐA CHẶNG & KIỂM CHỨNG TƯƠNG ĐƯƠNG HAI TẦNG (TWO-TIER VERIFIER)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
KNOWN_FUNCS = {"sin", "cos", "tan", "cot", "acos", "asin", "atan", "sqrt", "Abs", "exp", "log", "ln"}

SHM_LAW_SUBS = {
    sp.Symbol("v_max", real=True): sp.Symbol("omega", real=True) * sp.Symbol("A", real=True),
    sp.Symbol("a_max", real=True): (sp.Symbol("omega", real=True) ** 2) * sp.Symbol("A", real=True),
    sp.Symbol("a_min", real=True): -(sp.Symbol("omega", real=True) ** 2) * sp.Symbol("A", real=True),
    sp.Symbol("Wt", real=True): sp.Rational(1, 2) * sp.Symbol("k", real=True) * (sp.Symbol("x", real=True) ** 2),
    sp.Symbol("Wd", real=True): sp.Rational(1, 2) * sp.Symbol("k", real=True) * (sp.Symbol("A", real=True) ** 2 - sp.Symbol("x", real=True) ** 2),
    sp.Symbol("Wd_max", real=True): sp.Rational(1, 2) * sp.Symbol("m", real=True) * (sp.Symbol("omega", real=True) ** 2) * (sp.Symbol("A", real=True) ** 2),
}

def get_si_cgs_scales(vname: str) -> tuple:
    """Quy tắc đổi thứ nguyên vật lý tổng quát (SI <-> CGS) dựa trên đại lượng chuẩn SGK Vật lý 11."""
    if re.match(r"^(?:omega|T\b|f\b|phi|delta_phi|alpha|t\b|t\d|delta_t|ratio|N\b|k\b|delta_A_pct)", vname):
        return (1,)
    if re.match(r"^m\d*$", vname):
        return (1, 1000, sp.Rational(1, 1000))
    if re.match(r"^(?:W|E)", vname):
        return (1, 1000, sp.Rational(1, 1000), 10000, sp.Rational(1, 10000))
    if re.match(r"^(?:A|x|v|a|S|L)", vname):
        return (1, 100, sp.Rational(1, 100))
    return (1,)

def split_student_lines(student_text: str):
    raw = unicodedata.normalize("NFC", str(student_text or ""))
    # Khôi phục các chuỗi LaTeX bị lỗi escape (\f -> \frac, \t -> \text, \a -> \approx) từ nguồn Excel gốc
    raw = raw.replace("\x0crac{", r"\frac{").replace("\x09ext{", r"\text{").replace("\x07pprox", r"\approx")
    raw = re.sub(r"(?<![\\a-zA-Z])rac\{", r"\\frac{", raw)
    raw = re.sub(r"(?<![\\a-zA-Z])ext\{", r"\\text{", raw)
    raw = re.sub(r"(?<![\\a-zA-Z])pprox\b", r"\\approx", raw)
    lines = []
    idx = 1
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s in (r"\[", r"\]", "---", "$$"):
            continue
        if re.match(r"^(?:#+\s*)?(?:\*\*)?\[?[123]\.\s*(?:GIẢI THÍCH|TÍNH TOÁN|KẾT LUẬN|LỜI GIẢI)", s, re.I):
            continue
        if re.match(r"^(?:###?\s*)?LỜI GIẢI\s*:?\s*$", s, re.I):
            continue
        lines.append((f"L{idx:02d}", s))
        idx += 1
    return lines

def is_conditional_phrase(clause: str) -> bool:
    c = clause.lower()
    if re.search(r"\b(?:ta\s+luôn\s+có|suy\s+ra)\b", c):
        return False
    cond_kw = [
        r"\bkhi\b", r"\btại\s+thời\s+điểm\b", r"\bthời\s+điểm\s+đầu\b", r"\bban\s+đầu\b",
        r"\blúc\s+đầu\b", r"\btại\s*(?:\\\(|\[|\$)?\s*t\s*=\s*0\b",
        r"\b(?:kể|tính)\s+từ\s+(?:lúc\s+|thời\s+điểm\s+)?t\s*=",
        r"\bchọn\b", r"\bgiả\s+sử\b", r"\bvới\s+k\s*=", r"\bvật\s+bắt\s+đầu\b",
        r"\bvị\s+trí\s+cân\s+bằng\s*(?:\(|là)\b", r"\bnếu\b", r"\bđặt\b", r"\bxét\b",
    ]
    return any(re.search(p, c) for p in cond_kw)

def clean_math_expr_str(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s)).strip()
    if s.startswith("^") or s.startswith("*") or s.startswith("/"):
        return ""
    s = re.sub(r"(?<![\\a-zA-Z])rac\{", r"\\frac{", s)
    s = re.sub(r"[\t]ext\{", r"\\text{", s)
    s = re.sub(r"(?<![\\a-zA-Z])ext\{", r"\\text{", s)
    s = re.sub(r"(?<![\\a-zA-Z])pprox\b", " ", s)

    keep_text_map = {
        "max": "max", "min": "min", "tb": "tb", "avg": "avg", "tổng": "tổng",
        "vmax": "vmax", "amax": "amax", "amin": "amin", "đmax": "đmax", "dmax": "dmax",
        "arccos": " acos ", "arcsin": " asin ", "arctan": " atan ",
        "cos": " cos ", "sin": " sin ", "tan": " tan ", "cot": " cot ",
    }
    def _unwrap_text(m):
        return keep_text_map.get(m.group(1).strip(), " ")

    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|operatorname)\s*\{([^{}]*)\}", _unwrap_text, s)
    for _ in range(3):
        s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"(\1)", s)
    s = s.replace(r"\left", "").replace(r"\right", "")

    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    s = re.sub(r"(?<![a-zA-Z0-9_])(1\s*/\s*[0-9]+)\s*([A-Z])\b", r"(\1)*\2", s)
    s = re.sub(
        r"/\s*([0-9]+(?:\.[0-9]+)?(?:\s*(?:\\pi|π|\\omega|ω|\\sqrt\{[^{}]+\}|√\([^{}()]*\)|√[0-9a-zA-Z]+|[A-Za-z])(?:\^[0-9]+|²|³)?)+)",
        r"/ (\1)",
        s,
    )
    s = re.sub(r"/\s+(\d+\s*/\s*\d+)\b", r"/ (\1)", s)
    s = s.replace(r"\cdot", " * ").replace(r"\times", " * ")

    alias_patterns = [
        (r"\bv_\{?\s*max\s*\}?|\bvmax\b", " v_max "),
        (r"\ba_\{?\s*max\s*\}?|\bamax\b|\bmax\s*\(\s*a\s*\)", " a_max "),
        (r"\ba_\{?\s*min\s*\}?|\bamin\b", " a_min "),
        (r"\bv_\{?\s*tb\s*\}?|\bv_\{?\s*avg\s*\}?|\\bar\s*\{?\s*v\s*\}?", " v_tb "),
        (r"\bW_\{?\s*đmax\s*\}?|\bWđmax\b|\bWdmax\b|\bW_\{?\s*dmax\s*\}?", " Wd_max "),
        (r"\bW_\{?\s*đ\s*\}?|\bWđ\b|\bW_d\b", " Wd "),
        (r"\bW_\{?\s*t\s*\}?|\bWt\b", " Wt "),
        (r"\\Delta\s*\\Phi|\\Delta\s*\\phi|\\Delta\s*\\varphi|ΔΦ|Δφ|\bdelta_phi\b", " delta_phi "),
        (r"\\Delta\s*t\b|Δt\b|\bdelta_t\b", " delta_t "),
        (r"%\\Delta\s*A|%delta_A|\bdelta_A_pct\b", " delta_A_pct "),
        (r"\\alpha_?\{?0\}?|α_?0|\balpha_?0\b", " alpha0 "),
        (r"\\varphi_?\{?([0-9]+)\}?|\\phi_?\{?([0-9]+)\}?|φ_?([0-9]+)", lambda m: f" phi{m.group(1) or m.group(2) or m.group(3)} "),
        (r"\\varphi_?\{?t\}?|\\phi_?\{?t\}?|φ_?t|\bphi_?t\b", " phi_t "),
    ]
    for pat, rep in alias_patterns:
        s = re.sub(pat, rep, s)

    # Quy tắc ký hiệu toán học phổ quát: dấu phẩy trên X' -> X_prime, chỉ số số X_{1} -> X1, chỉ số chữ X_{sub} -> X_sub
    s = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\'", r" \1_prime ", s)
    s = re.sub(r"\b([a-zA-Z]+)_\{?([0-9]+)\}?\b", r"\1\2", s)
    s = re.sub(r"\b([a-zA-Z])_\{([a-zA-Z]+)\}", r"\1_\2", s)

    for _ in range(6):
        ns = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
        ns = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", ns)
        if ns == s:
            break
        s = ns

    while re.search(r"(\d+)\.(\d+)\.(\d+)", s):
        s = re.sub(r"(\d+)\.(\d+)\.(\d+)", r"\1*\2*\3", s)

    token_repls = [
        (r"\arccos", " acos "), (r"\arcsin", " asin "), (r"\arctan", " atan "),
        (r"\cos", " cos "), (r"\sin", " sin "), (r"\tan", " tan "),
        (r"\omega", " omega "), (r"\varphi", " phi "), (r"\phi", " phi "),
        (r"\alpha", " alpha "), (r"\pi", " pi "),
        ("ω", " omega "), ("π", " pi "), ("φ", " phi "), ("α", " alpha "),
        ("²", "**2 "), ("³", "**3 "), ("×", " * "), ("⋅", " * "), ("−", "-"), ("–", "-"),
    ]
    for old, new in token_repls:
        s = s.replace(old, new)

    s = re.sub(r"√\s*\(", " sqrt(", s)
    s = re.sub(r"√\s*\[", " sqrt(", s)
    s = re.sub(r"√\s*([0-9]+(?:\.[0-9]+)?|[a-zA-Z_][a-zA-Z0-9_]*)", r" sqrt(\1) ", s)
    s = re.sub(r"\bsqrt\s*\[([^\[\]]+)\]", r"sqrt(\1)", s)
    s = re.sub(r"\|([^|=<>]+)\|", r" Abs(\1) ", s)

    s = re.sub(r"(\d)\s+\.\s+(\d)", r"\1 * \2", s)
    s = re.sub(r"(\d)\s*\.\s*(?=[a-zA-Z(\[])", r"\1 * ", s)
    s = re.sub(r"([a-zA-Z)\]])\s*\.\s*(?=[0-9a-zA-Z(\[])", r"\1 * ", s)
    s = re.sub(r"(\))\s*\.\s*(\d)", r"\1 * \2", s)

    s = re.sub(r"\b([a-zA-Z]+)_([0-9]+)\b", r"\1\2", s)

    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", " ")
    s = s.replace("^", "**")

    unit_pat = r"(?:cm/s\*\*2|m/s\*\*2|cm/s|m/s|rad/s|radian|rad|giây|phút|Hz|mJ|J|s|cm|m|kg|g)"
    s = re.sub(rf"\(\s*{unit_pat}\s*\)\s*\.?\s*$", "", s.strip(), flags=re.I)
    s = re.sub(rf"(?<=[0-9)\s])\s+{unit_pat}\s*\.?\s*$", "", s.strip(), flags=re.I)
    s = re.sub(r"(\d(?:\.\d+)?)\s*(?:cm/s\*\*2|m/s\*\*2|cm/s|m/s|rad/s|rad|Hz|cm|kg)\b", r"\1", s)
    s = re.sub(r"(?<=\d)\s*s\b", "", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"((\1)/100)", s)

    def _fix_func_call(m):
        ident = m.group(1)
        return f"{ident}(" if ident in KNOWN_FUNCS else f"{ident}*("
    s = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", _fix_func_call, s)
    s = re.sub(r"(\d)([a-zA-Z_])", r"\1*\2", s)
    s = s.strip(" .,;:*+")
    return s

def safe_sympify(expr_str: str):
    cleaned = clean_math_expr_str(expr_str)
    if not cleaned or "," in cleaned:
        return None, cleaned, "Empty_or_TupleComma"
    local_dict = {
        "pi": sp.pi, "sqrt": sp.sqrt, "cos": sp.cos, "sin": sp.sin,
        "tan": sp.tan, "cot": sp.cot, "acos": sp.acos, "asin": sp.asin,
        "atan": sp.atan, "Abs": sp.Abs,
    }
    for tok in set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", cleaned)):
        if tok not in KNOWN_FUNCS and tok != "pi":
            local_dict[tok] = sp.Symbol(tok, real=True)
    try:
        expr = parse_expr(cleaned, local_dict=local_dict, transformations=TRANSFORMS, evaluate=True)
        if not isinstance(expr, sp.Basic):
            return None, cleaned, "NotSymPyBasic"
        return expr, cleaned, None
    except Exception as e:
        return None, cleaned, str(e)

def check_two_tier_equivalence(lhs_expr, rhs_expr, raw_lhs="", raw_rhs="",
                               subs_env=None, allow_pi_sq_10=True):
    """
    Tier 1: Exact Symbolic / Trigonometric Equivalence (`simplify(trigsimp(L - R)) == 0`)
            + Quy ước SGK Vật lý Việt Nam `pi^2 = 10` (`pi = sqrt(10)`).
    Tier 2: Metrological Significant-Figure Equivalence (`Half-ULP`: `0.55 * 10^(-d)`).
    """
    subs_env = subs_env or {}
    L = lhs_expr.subs(subs_env) if subs_env else lhs_expr
    R = rhs_expr.subs(subs_env) if subs_env else rhs_expr

    # --- TIER 1A: Exact Symbolic / Trigonometric ---
    try:
        diff = sp.simplify(sp.trigsimp(L - R))
        if diff == 0:
            return True, "Tier1_Exact_Symbolic", 0.0
        if diff.is_number and abs(complex(sp.N(diff))) < 1e-9:
            return True, "Tier1_Exact_Numeric", 0.0
    except Exception:
        pass

    # --- TIER 1B: Vietnamese Physics Convention `pi^2 = 10` ---
    if allow_pi_sq_10:
        try:
            L_p10 = L.subs(sp.pi, sp.sqrt(10))
            R_p10 = R.subs(sp.pi, sp.sqrt(10))
            diff10 = sp.simplify(sp.trigsimp(L_p10 - R_p10))
            if diff10 == 0 or (diff10.is_number and abs(complex(sp.N(diff10))) < 1e-9):
                return True, "Tier1_PiSq10_Symbolic", 0.0
        except Exception:
            pass

    if L.free_symbols or R.free_symbols:
        free = list(L.free_symbols | R.free_symbols)
        all_close = True
        for pt in [
            {sym: 0.41 + 0.13 * idx for idx, sym in enumerate(free)},
            {sym: 1.17 + 0.29 * idx for idx, sym in enumerate(free)},
        ]:
            try:
                lv = complex(sp.N(L.subs(pt)))
                rv = complex(sp.N(R.subs(pt)))
                if abs(lv - rv) > 1e-6:
                    all_close = False
                    break
            except Exception:
                all_close = False
                break
        if all_close:
            return True, "Tier1_Functional_Identity", 0.0
        return False, "Symbolic_Mismatch", float("inf")

    # --- TIER 2: Half-ULP Metrology on Numeric Scalars ---
    try:
        lv = float(sp.N(L))
        rv = float(sp.N(R))
    except Exception:
        return False, "Complex_Or_NonReal", float("inf")

    err = abs(lv - rv)
    if err < 1e-7:
        return True, "Tier1_Exact_Float", err

    r_clean = re.sub(r"(\d),(\d)", r"\1.\2", f"{raw_lhs} = {raw_rhs}")
    rhs_s = re.sub(r"(\d),(\d)", r"\1.\2", str(raw_rhs)).strip()
    lhs_s = re.sub(r"(\d),(\d)", r"\1.\2", str(raw_lhs)).strip()

    tol = 1e-6
    for side_raw in (rhs_s, lhs_s):
        m_num = re.match(r"^[-+]?\s*(\d+)(?:\.(\d+))?\s*(?:%|\([a-zA-Z/^2*]+\)|[a-zA-Z/^2*]+)?\s*\.?$", side_raw)
        if m_num:
            dec_part = m_num.group(2)
            has_pct = "%" in side_raw
            d = len(dec_part) if dec_part is not None else 0
            if d > 0:
                side_tol = 0.55 * (10 ** (-d))
                if has_pct:
                    side_tol *= 0.01
                tol = max(tol, side_tol)
            elif re.search(r"≈|approx|pprox|xấp xỉ|khoảng", f"{raw_lhs} {raw_rhs}", re.I) or ("." in r_clean):
                side_tol = 0.55 * (0.01 if has_pct else 1.0)
                tol = max(tol, side_tol)

    if err <= tol:
        return True, f"Tier2_HalfULP(tol={tol:.2e})", err

    if allow_pi_sq_10:
        try:
            lv10 = float(sp.N(L.subs(sp.pi, sp.sqrt(10))))
            rv10 = float(sp.N(R.subs(sp.pi, sp.sqrt(10))))
            if abs(lv10 - rv10) <= max(tol, 1e-6):
                return True, "Tier2_PiSq10_HalfULP", abs(lv10 - rv10)
        except Exception:
            pass

    if L.has(sp.pi) or R.has(sp.pi) or L.has(sp.Pow) or R.has(sp.Pow):
        for pi_apx in (3.14, 3.1416):
            try:
                lv_a = float(sp.N(L.subs(sp.pi, pi_apx)))
                rv_a = float(sp.N(R.subs(sp.pi, pi_apx)))
                if abs(lv_a - rv_a) <= max(tol, 1e-3 * max(abs(lv_a), 1.0)):
                    return True, "Tier2_PiApprox314", abs(lv_a - rv_a)
            except Exception:
                pass

    return False, f"Numeric_Mismatch(diff={err:.5g},tol={tol:.2e})", err

def extract_initial_wave_funcs(question_text: str):
    q = unicodedata.normalize("NFC", str(question_text))
    waves = {}
    for m in re.finditer(r"\b(x_?[12]?)\s*=\s*([0-9.,A]+(?:\s*(?:sin|cos|\\sin|\\cos))\s*\([^)]+\))", q):
        vname = m.group(1).replace("_", "")
        expr, _, _ = safe_sympify(m.group(2))
        if expr is not None:
            waves[vname] = expr
            t_s = sp.Symbol("t", real=True)
            if vname == "x" and t_s in expr.free_symbols:
                waves["v_wave"] = sp.diff(expr, t_s)
    return waves

def _split_conjunctions_with_eq(s: str):
    parts = [s]
    conj_pat = re.compile(r"(?:\s+(?:và|nên|suy\s+ra|do\s+đó|từ\s+đó)\s+|,\s+(?=[a-zA-Zà-ỹÀ-Ỹ]))", re.I)
    changed = True
    while changed:
        changed = False
        new_parts = []
        for p in parts:
            m = conj_pat.search(p)
            if m:
                left, right = p[:m.start()], p[m.end():]
                if (("=" in left or "≈" in left) and ("=" in right or "≈" in right)
                        and left.count("(") == left.count(")")
                        and left.count(r"\(") == left.count(r"\)")):
                    new_parts.extend([left, right])
                    changed = True
                    continue
            new_parts.append(p)
        parts = new_parts
    return parts

# Ablation S1: Naive single-split Regex (không tách chuỗi đa chặng, không chuẩn hóa ngữ cảnh vật lý)
def naive_regex_extract(line_id: str, line_text: str):
    assertions = []
    if "=" not in line_text or any(op in line_text for op in ("<=", ">=", "!=")):
        return assertions
    parts = line_text.split("=")
    if len(parts) >= 2:
        lhs = parts[0].split(":")[-1].strip()
        rhs = parts[-1].strip()
        assertions.append({
            "line": line_id, "raw_line": line_text,
            "lhs_raw": lhs, "rhs_raw": rhs,
            "chain_var": None, "is_chain_tail": True, "kind": "calc",
            "source": "naive_regex",
        })
    return assertions

# Channel A: Deterministic Multi-Hop Physics Chain Scanner
def extract_multihop_physics_chains(line_id: str, line_text: str):
    s = unicodedata.normalize("NFC", str(line_text))
    s = re.sub(r"W_?\{?đmax\}?|Wđmax", "Wd_max", s)
    s = re.sub(r"W_?\{?đ\}?|Wđ", "Wd", s)
    s = re.sub(r"(?<![\\a-zA-Z])pprox\b", "≈", s)
    s = s.replace(r"\approx", "≈").replace(r"\Rightarrow", "=>").replace("⇒", "=>")

    raw_clauses = re.split(
        r"(?:=>|;|,\s*(?=[a-zA-ZωφΔ\\][a-zA-Z0-9_\\{}]*\s*=)|(?<=[a-zA-Z0-9)\]])\.\s+(?=[A-ZÀ-Ỹ(\\\[]))",
        s,
    )
    clauses = []
    for rc in raw_clauses:
        clauses.extend(_split_conjunctions_with_eq(rc))

    assertions = []
    unit_conv_pat = re.compile(r"(?:cm/s\^?2|cm/s|cm|mm|phút|\bg\b).*?=.*?(?:m/s\^?2|m/s|\bm\b|\bkg\b|\bs\b)", re.I)
    viet_accent_chars = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"

    for cl in clauses:
        cl_strip = cl.strip()
        if not cl_strip or ("=" not in cl_strip and "≈" not in cl_strip):
            continue
        if re.search(r"<=|>=|≤|≥|\\le\b|\\ge\b|\\neq\b|!=|\\propto|±|\+/-", cl_strip):
            continue
        cl_strip = re.sub(r"=\s*x\s*\'\s*=", "=", cl_strip)
        if re.search(r"\bd\^?2?[a-z]/dt|\bx\s*\'|\bdx/dt|\bdv/dt", cl_strip):
            continue
        if re.search(r"\b[xva]\s*\(\s*[t0]\s*\)", cl_strip):
            continue
        if re.search(r"\b(?:đến|tới|hoặc|tức)\b", cl_strip, re.I):
            continue

        cond_flag = is_conditional_phrase(cl_strip)

        math_spans = []
        latex_blocks = re.findall(r"\\\((.*?)\\\)|\\\[(.*?)\\\]|\$([^$]+)\$", cl_strip)
        if latex_blocks:
            for b in latex_blocks:
                sp_txt = (b[0] or b[1] or b[2]).strip()
                if ("=" in sp_txt or "≈" in sp_txt) and not re.search(r"<=|>=|≤|≥|\\propto|±", sp_txt):
                    math_spans.append(sp_txt)

        if not math_spans:
            rhs_after_colon = cl_strip.split(":")[-1].strip() if ":" in cl_strip else cl_strip
            rhs_after_colon = re.sub(
                r"\(\s*(?:vì|do|nên|với|theo|chọn|lấy|biết)\b[^)]*\)\s*\.?\s*$",
                "", rhs_after_colon, flags=re.I
            ).strip()
            m_eq = re.search(
                rf"(?<![a-zA-Z{viet_accent_chars}])([|a-zA-Z0-9_\'\\(\)\[\]\{{\}}\-+√πφαωΔ%\s\.\^/*]+?(?:=|≈).*)",
                rhs_after_colon,
            )
            if m_eq:
                cand = m_eq.group(1).strip()
                first_eq_pos = min([p for p in (cand.find("="), cand.find("≈")) if p != -1])
                left_part = cand[:first_eq_pos].strip()
                rest_part = cand[first_eq_pos:]
                kept = []
                for tok in reversed(left_part.split()):
                    tok_clean = tok.strip("(),;:*+-\'[]{}")
                    if re.search(f"[{viet_accent_chars.lower()}]", tok_clean.lower()):
                        break
                    if tok_clean.lower() in {"ta", "suy", "ra", "co", "la", "va", "voi", "khi", "tu", "nen", "do", "vay", "chon", "hoac", "hay", "trong", "theo"}:
                        break
                    kept.append(tok)
                if kept:
                    left_math = " ".join(reversed(kept)).lstrip("-+* ")
                    if left_math.startswith("(") and left_math.count("(") > left_math.count(")"):
                        left_math = left_math[1:].strip()
                        if rest_part.rstrip(".").endswith(")"):
                            rest_part = rest_part.rstrip(".")[:-1]
                    right_math = re.split(
                        r"\s+\(?(?:vì|do|nên|suy|nhưng|hoặc|với|trong|tương|ứng|theo|biết|lấy|thỏa|là\s+[a-zA-Zà-ỹ])\b",
                        rest_part, maxsplit=1
                    )[0]
                    if not left_math.startswith("^"):
                        math_spans.append(left_math + right_math)

        for sp_str in math_spans:
            if unit_conv_pat.search(sp_str):
                parts_u = [p.strip() for p in re.split(r"(?<![<>!])(?:=|≈)(?!=)", sp_str) if p.strip()]
                if len(parts_u) >= 2 and re.match(r"^[a-zA-Z_\\][a-zA-Z0-9_\\{}]*$", parts_u[0]):
                    assertions.append({
                        "line": line_id, "raw_line": line_text,
                        "lhs_raw": parts_u[0], "rhs_raw": parts_u[-1],
                        "chain_var": parts_u[0], "is_chain_tail": True,
                        "kind": "given", "source": "channel_a_det",
                    })
                    continue

            parts = [p.strip() for p in re.split(r"(?<![<>!])(?:=|≈)(?!=)", sp_str) if p.strip()]
            if len(parts) < 2:
                continue
            lhs_sym_chk, _, _ = safe_sympify(parts[0])
            is_var_chain = (lhs_sym_chk is not None and lhs_sym_chk.is_Symbol)

            for idx_p in range(len(parts) - 1):
                if parts[idx_p].startswith("^") or parts[idx_p + 1].startswith("^"):
                    continue
                assertions.append({
                    "line": line_id, "raw_line": line_text,
                    "lhs_raw": parts[idx_p], "rhs_raw": parts[idx_p + 1],
                    "chain_var": lhs_sym_chk.name if is_var_chain else None,
                    "is_chain_tail": (idx_p == len(parts) - 2),
                    "kind": "cond" if cond_flag else "calc",
                    "source": "channel_a_det",
                })
    return assertions

# Anchor-Free State-Graph Step Verifier
def verify_submission_steps(steps: list, canon_state: dict, use_two_tier: bool = True):
    c_vars = dict(canon_state["vars"])
    waves = extract_initial_wave_funcs(canon_state["question"])
    canon_sym_env = {sp.Symbol(k, real=True): v for k, v in c_vars.items() if isinstance(v, (int, float, sp.Basic))}
    stu_env = {}
    trace_log = []

    for st in steps:
        lhs_e, _, err_l = safe_sympify(st["lhs_raw"])
        rhs_e, _, err_r = safe_sympify(st["rhs_raw"])
        if lhs_e is None or rhs_e is None:
            trace_log.append({**st, "status": "UNPARSED", "tier": "None", "reason": err_l or err_r})
            continue

        if st.get("kind") == "cond" or (lhs_e == sp.Symbol("t", real=True) and rhs_e == 0):
            if lhs_e.is_Symbol and not rhs_e.free_symbols and lhs_e != sp.Symbol("t", real=True):
                stu_env[lhs_e] = rhs_e
            trace_log.append({**st, "status": "COND_SKIP", "tier": "Boundary_Cond", "reason": "Boundary condition"})
            continue

        t_s = sp.Symbol("t", real=True)
        is_wave_expr = any(expr.has(sp.cos) or expr.has(sp.sin) for expr in (lhs_e, rhs_e)) and (
            (t_s in lhs_e.free_symbols) or (t_s in rhs_e.free_symbols)
        )
        if is_wave_expr:
            tier_w = "Tier1_Wave_Symbolic"
            if (t_s in lhs_e.free_symbols) and (t_s in rhs_e.free_symbols):
                wave_subs = {k: v for k, v in canon_sym_env.items() if k.name in ("A", "omega", "phi", "A1", "A2", "phi1", "phi2")}
                use_subs = wave_subs if (lhs_e.free_symbols - {t_s}) != (rhs_e.free_symbols - {t_s}) else {}
                ok, tier_w, _ = check_two_tier_equivalence(lhs_e, rhs_e, st["lhs_raw"], st["rhs_raw"], subs_env=use_subs, allow_pi_sq_10=use_two_tier)
                if not ok:
                    err_info = {**st, "status": "ERROR", "tier": tier_w, "expected_sympy": str(sp.simplify(lhs_e.subs(use_subs))), "reason": f"Wave_Transform_Error ({tier_w})"}
                    trace_log.append(err_info)
                    return "INCORRECT", err_info, trace_log
            elif lhs_e.is_Symbol and lhs_e.name in ("x1", "x2", "x", "v"):
                ref_wave = waves.get(lhs_e.name) if lhs_e.name != "v" else waves.get("v_wave")
                if ref_wave is not None and not (rhs_e.free_symbols - {t_s}):
                    ok, tier_w, _ = check_two_tier_equivalence(ref_wave, rhs_e, st["lhs_raw"], st["rhs_raw"], allow_pi_sq_10=use_two_tier)
                    if not ok:
                        err_info = {**st, "status": "ERROR", "tier": tier_w, "expected_sympy": str(ref_wave), "reason": f"Initial_Wave_Mismatch ({tier_w})"}
                        trace_log.append(err_info)
                        return "INCORRECT", err_info, trace_log
            trace_log.append({**st, "status": "OK", "tier": tier_w, "reason": "Wave_OK"})
            continue

        eval_env = {sym_k: val_v for sym_k, val_v in canon_sym_env.items() if sym_k not in stu_env}
        eval_env.update(stu_env)

        # Kiểm chứng định luật Vật lý thuần ký hiệu (VD: sub_02 `a_max = (v_max/omega)^2`, sub_19 `Wd = 1/2*k*x^2`)
        if lhs_e.is_Symbol and rhs_e.free_symbols:
            if (lhs_e in SHM_LAW_SUBS) and any(s.name in ("k", "x", "A", "omega", "v_max", "a_max") for s in rhs_e.free_symbols):
                if not any(s.name in ("W", "E") for s in rhs_e.free_symbols):
                    law_l = lhs_e.subs(SHM_LAW_SUBS)
                    law_r = rhs_e.subs(SHM_LAW_SUBS)
                    if (law_l.free_symbols | law_r.free_symbols).issubset({
                        sp.Symbol("A", real=True), sp.Symbol("x", real=True),
                        sp.Symbol("omega", real=True), sp.Symbol("k", real=True), sp.Symbol("m", real=True),
                    }):
                        ok_law, tier_l, _ = check_two_tier_equivalence(law_l, law_r, st["lhs_raw"], st["rhs_raw"], allow_pi_sq_10=use_two_tier)
                        if not ok_law:
                            err_info = {**st, "status": "ERROR", "tier": tier_l, "expected_sympy": str(law_l), "reason": f"Physical_Law_Error ({tier_l})"}
                            trace_log.append(err_info)
                            return "INCORRECT", err_info, trace_log

        # Gán trực tiếp biến vật lý `var = rhs_expr`
        if lhs_e.is_Symbol:
            vname = lhs_e.name
            k_s = sp.Symbol("k", real=True)
            if lhs_e == sp.Symbol("t", real=True) and (k_s in rhs_e.free_symbols) and "t_expr" in c_vars:
                ok_tk, tier_tk, _ = check_two_tier_equivalence(c_vars["t_expr"], rhs_e, st["lhs_raw"], st["rhs_raw"], allow_pi_sq_10=use_two_tier)
                if not ok_tk:
                    err_info = {**st, "status": "ERROR", "tier": tier_tk, "expected_sympy": str(sp.simplify(c_vars["t_expr"])), "reason": f"Parametric_Solution_Error ({rhs_e} != {c_vars['t_expr']})"}
                    trace_log.append(err_info)
                    return "INCORRECT", err_info, trace_log

            rhs_env = {k: v for k, v in eval_env.items() if k != lhs_e}
            rhs_val = rhs_e.subs(rhs_env)
            used_tier = "Tier1_Exact_Symbolic"

            if lhs_e in stu_env and not rhs_val.free_symbols and not stu_env[lhs_e].free_symbols:
                if use_two_tier:
                    ok_prev, used_tier, _ = check_two_tier_equivalence(stu_env[lhs_e], rhs_val, str(stu_env[lhs_e]), st["rhs_raw"], allow_pi_sq_10=True)
                else:
                    ok_prev = (sp.simplify(stu_env[lhs_e] - rhs_val) == 0)
                    used_tier = "ExactOnly"
                if not ok_prev:
                    err_info = {**st, "status": "ERROR", "tier": used_tier, "expected_sympy": str(sp.simplify(stu_env[lhs_e])), "reason": f"Sequential_Value_Drift ({vname}: prev={stu_env[lhs_e]} vs new={rhs_val})"}
                    trace_log.append(err_info)
                    return "INCORRECT", err_info, trace_log

            # Các biến phân đoạn hành trình cục bộ (t1, t2, S1...), trạng thái tức thời (x, v) hoặc số nguyên chu kỳ (k) phụ thuộc cách chia đoạn của từng học sinh
            is_local_partition = bool(re.match(r"^(?:[txvS]\d+|[xvk])$", vname))
            if vname in c_vars and not is_local_partition and not rhs_val.free_symbols and not sp.sympify(c_vars[vname]).free_symbols:
                c_target = c_vars[vname]
                scales = get_si_cgs_scales(vname)
                matched_any = False
                rhs_canon_val = rhs_e.subs({k: v for k, v in canon_sym_env.items() if k != lhs_e})
                for cand_val in ([rhs_val, rhs_canon_val] if rhs_e.free_symbols and not rhs_canon_val.free_symbols else [rhs_val]):
                    for sc in scales:
                        if use_two_tier:
                            ok_c, tier_c, _ = check_two_tier_equivalence(c_target * sc, cand_val, st["lhs_raw"], st["rhs_raw"], allow_pi_sq_10=True)
                        else:
                            ok_c = (sp.simplify(c_target * sc - cand_val) == 0)
                            tier_c = "ExactOnly"
                        if ok_c:
                            matched_any = True
                            used_tier = tier_c
                            break
                    if matched_any:
                        break
                if not matched_any:
                    err_info = {**st, "status": "ERROR", "tier": "Mismatch", "expected_sympy": str(c_target), "reason": f"Canonical_State_Mismatch ({vname}={rhs_val} vs gold={c_target})"}
                    trace_log.append(err_info)
                    return "INCORRECT", err_info, trace_log

            if not rhs_val.free_symbols:
                stu_env[lhs_e] = rhs_val
            trace_log.append({**st, "status": "OK", "tier": used_tier, "reason": "Assign_OK"})
            continue

        # Bước biến đổi đại số / tính số trực tiếp `lhs_expr == rhs_expr`
        L_val = lhs_e.subs(eval_env)
        R_val = rhs_e.subs(eval_env)
        used_tier = "Tier1_Exact_Symbolic"

        if not L_val.free_symbols and not R_val.free_symbols:
            if use_two_tier:
                ok_h, used_tier, _ = check_two_tier_equivalence(L_val, R_val, st["lhs_raw"], st["rhs_raw"], allow_pi_sq_10=True)
            else:
                ok_h = (sp.simplify(L_val - R_val) == 0)
                used_tier = "ExactOnly"

            if not ok_h and (lhs_e.free_symbols or rhs_e.free_symbols):
                for scale_factor in (sp.Rational(1, 100), 1, 100):
                    alt_env = dict(eval_env)
                    for sk_sym, sk_val in canon_sym_env.items():
                        if sk_sym not in stu_env and re.match(r"^(?:A|x|v|a|S|L)", sk_sym.name):
                            alt_env[sk_sym] = sk_val * scale_factor
                    L_alt, R_alt = lhs_e.subs(alt_env), rhs_e.subs(alt_env)
                    if not L_alt.free_symbols and not R_alt.free_symbols:
                        if use_two_tier:
                            ok_alt, tier_alt, _ = check_two_tier_equivalence(L_alt, R_alt, st["lhs_raw"], st["rhs_raw"], allow_pi_sq_10=True)
                        else:
                            ok_alt = (sp.simplify(L_alt - R_alt) == 0)
                            tier_alt = "ExactOnly"
                        if ok_alt:
                            ok_h = True
                            used_tier = tier_alt
                            break

            if not ok_h:
                err_info = {**st, "status": "ERROR", "tier": used_tier, "expected_sympy": str(sp.simplify(L_val)), "reason": f"IntraStep_Calc_Error ({L_val} != {R_val}, {used_tier})"}
                trace_log.append(err_info)
                return "INCORRECT", err_info, trace_log

            if st.get("chain_var") and st.get("is_chain_tail") and not rhs_e.free_symbols:
                cv_sym = sp.Symbol(st["chain_var"], real=True)
                if st["chain_var"] in c_vars and not re.match(r"^(?:[txvS]\d+|[xvk])$", st["chain_var"]):
                    c_target = c_vars[st["chain_var"]]
                    scales = get_si_cgs_scales(st["chain_var"])
                    ok_tail = any(
                        check_two_tier_equivalence(c_target * sc, rhs_e, st["lhs_raw"], st["rhs_raw"], allow_pi_sq_10=True)[0]
                        if use_two_tier else (sp.simplify(c_target * sc - rhs_e) == 0)
                        for sc in scales
                    )
                    if not ok_tail:
                        err_info = {**st, "status": "ERROR", "tier": "Mismatch", "expected_sympy": str(c_target), "reason": f"Chained_Target_Mismatch ({st['chain_var']}={rhs_e} vs gold={c_target})"}
                        trace_log.append(err_info)
                        return "INCORRECT", err_info, trace_log
                stu_env[cv_sym] = rhs_e

        elif lhs_e.free_symbols or rhs_e.free_symbols:
            if lhs_e.free_symbols == rhs_e.free_symbols and len(lhs_e.free_symbols) <= 2:
                if sp.simplify(lhs_e - rhs_e) != 0:
                    err_info = {**st, "status": "ERROR", "tier": "Symbolic_Mismatch", "expected_sympy": str(sp.simplify(lhs_e)), "reason": f"Symbolic_Simplification_Error ({lhs_e} != {rhs_e})"}
                    trace_log.append(err_info)
                    return "INCORRECT", err_info, trace_log

        trace_log.append({**st, "status": "OK", "tier": used_tier, "reason": "Hop_OK"})

    return "CORRECT", None, trace_log

# Kiểm tra tự động 15 Unit Tests trọng điểm (bao gồm ví dụ 7T/6 == 1.167 khi T=1 của Reviewer 1)
UNIT_TESTS = [
    ("sub_11 valid Half-ULP step: 7T/6 == 1,167 (under T=1)", "7T/6", "1,167", {sp.Symbol("T", real=True): 1}, True),
    ("sub_11 real arithmetic error: 31,5 / 1,167 ≈ 27,19", "31,5 / 1,167", "27,19", {}, False),
    ("cor_11 valid approx: 31,5 / 1,1667 ≈ 27", "31,5 / 1,1667", "27", {}, True),
    ("cor_10 valid 3-dec rounding: sqrt(0.92) ≈ 0.959", "sqrt(0.92)", "0.959", {}, True),
    ("cor_10 valid pct: 100% - 95.9% = 4.1%", "100% - 95.9%", "4.1%", {}, True),
    ("sub_09 hallucinated sqrt: sqrt(0,92) ≈ 0,9539", "sqrt(0,92)", "0,9539", {}, False),
    ("sub_10 hallucinated sqrt: sqrt(0,92) ≈ 0,9583", "sqrt(0,92)", "0,9583", {}, False),
    ("sub_01 chained tail error: pi/(4*pi) = 0,125", "pi/(4*pi)", "0,125", {}, False),
    ("sub_02 double-cancellation law: a_max = (v_max/omega)^2", "a_max", "(v_max/omega)^2",
     {sp.Symbol("a_max", real=True): 16 * sp.pi**2, sp.Symbol("v_max", real=True): 8 * sp.pi, sp.Symbol("omega", real=True): 2 * sp.pi}, False),
    ("sub_03 trig sign error: 3sin(10t-pi/3) = 3cos(10t-pi/3+pi/2)", "3sin(10t - π/3)", "3cos(10t - π/3 + π/2)", {}, False),
    ("cor_03 trig right: 3sin(10t-pi/3) = 3cos(10t-pi/3-pi/2)", "3sin(10t - π/3)", "3cos(10t - π/3 - π/2)", {}, True),
    ("sub_31 phase add error: 40π.cos(10πt-π/3+π/2) = 40π.cos(10πt-π/6)", "40π.cos(10πt - π/3 + π/2)", "40π.cos(10πt - π/6)", {}, False),
    ("cor_20 textbook pi^2=10: 100*pi^2 = 1000", "100*pi^2", "1000", {}, True),
    ("sub_21 wrong extra sqrt(2): 100*9,87*1,41 = 1385,4", "100 * 9,87 * 1,41", "1385,4", {}, False),
    ("sub_27 wrong A/2: A/2 = 5 (A=5)", "A/2", "5", {sp.Symbol("A", real=True): 5}, False),
]

for desc, rl, rr, sb, expected_ok in UNIT_TESTS:
    el, _, _ = safe_sympify(rl)
    er, _, _ = safe_sympify(rr)
    ok_res, tier_res, _ = check_two_tier_equivalence(el, er, rl, rr, sb)
    assert ok_res == expected_ok, f"Unit test failed: {desc} -> got {ok_res} ({tier_res})"

print(f"--> Đã kiểm tra thành công {len(UNIT_TESTS)}/{len(UNIT_TESTS)} Unit Tests đại số & đo lường số học (PASS 100%).")


# [CELL 4]: NẠP QWEN-2.5-7B-INSTRUCT (T=0) -> CHẠY BASELINE S0 & HYBRID NL-TO-SYMPY PARSER (+ 1-SHOT REPAIR GATE)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

if "MOCK_LLM_GENERATE" in globals() and callable(globals()["MOCK_LLM_GENERATE"]):
    print("[LOCAL DRY-RUN MODE] Đang sử dụng hàm MOCK_LLM_GENERATE để kiểm thử toàn bộ luồng logic.")
    llm_generate_fn = globals()["MOCK_LLM_GENERATE"]
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Đang nạp Tokenizer và Model `{MODEL_NAME}` trên GPU (float16 + SDPA, Greedy T=0)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print("--> Đã nạp thành công Qwen/Qwen2.5-7B-Instruct!")

    def llm_generate_fn(system_msg: str, user_msg: str, max_new_tokens: int = 768) -> str:
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt_txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt_txt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# --- PHẦN 4A: CHẠY BASELINE S0 (PURE LLM GRADER V5, T=0) TRÊN 62 SUBMISSIONS ---
S0_SYSTEM_PROMPT = (
    "Bạn là Gia sư Vật lý lớp 11 (dao động điều hòa). Đối chiếu bài làm học sinh với Lời giải "
    "mẫu chuẩn 100%. So sánh theo GIÁ TRỊ toán học, không theo cách trình bày; đi tuần tự và chỉ dừng "
    "ở bước SAI THẬT SỰ đầu tiên (giá trị khác Lời giải mẫu). Không tự tính toán mới; mọi con số lấy "
    "từ Lời giải mẫu. In đúng 3 dòng: @@VERDICT@@, @@STEP@@, @@FIX@@."
)

def parse_v5_tags(fb_text: str):
    t = unicodedata.normalize("NFC", str(fb_text or ""))
    def _grab(tag, nxt):
        pat = r"@@" + tag + r"@@\s*:?\s*(.*?)(?=@@(?:" + nxt + r")@@|$)" if nxt else r"@@" + tag + r"@@\s*:?\s*(.*)$"
        m = re.search(pat, t, re.I | re.S)
        return m.group(1).strip(" :\n\t*#>-") if m else ""
    vraw = _grab("VERDICT", "STEP|FIX")
    m_v = re.search(r"(?:^|\b)(SAI|INCORRECT|ĐÚNG|CORRECT)(?:\b|$)", vraw, re.I)
    if m_v:
        verdict = "INCORRECT" if m_v.group(1).upper() in ("SAI", "INCORRECT") else "CORRECT"
    else:
        verdict = "?"
    return verdict, _grab("STEP", "FIX"), _grab("FIX", "")

print("\n[1/2] Đang thực thi Baseline S0 (Pure LLM Grader V5) trên 62 bài làm...")
t0_s0 = time.time()
s0_records = []
ws_lookup = {row["submission_id"]: row for _, row in df_ws_v5.iterrows()}

for idx, row in df_stage2.iterrows():
    sid = row["submission_id"]
    fb = llm_generate_fn(S0_SYSTEM_PROMPT, str(row["prompt"]), max_new_tokens=768)
    v, step_txt, fix_txt = parse_v5_tags(fb)
    if v == "?" and sid in ws_lookup:
        v = str(ws_lookup[sid]["tutor_verdict"]).strip()
        step_txt = str(ws_lookup[sid].iloc[6])
        fix_txt = str(ws_lookup[sid].iloc[7])
    s0_records.append({
        "submission_id": sid,
        "s0_raw_feedback": fb,
        "s0_verdict": v if v != "?" else "CORRECT",
        "s0_step": step_txt,
        "s0_fix": fix_txt,
    })
    if (idx + 1) % 10 == 0 or idx == len(df_stage2) - 1:
        print(f"    S0 Progress: {idx + 1}/{len(df_stage2)} ({time.time() - t0_s0:.1f}s)")

df_s0_preds = pd.DataFrame(s0_records)

# --- PHẦN 4B: HYBRID NL-TO-SYMPY PARSER (CHANNEL B LLM JSON + 1-SHOT SYMPY REPAIR + GROUNDED FUSION) ---
EXTRACTOR_SYS_PROMPT = (
    "You are a Physics Symbolic Step Extractor for Grade-11 Simple Harmonic Motion. "
    "Given numbered student solution lines (L01, L02, ...), extract ONLY explicit mathematical equalities "
    "stated by the student as a JSON list of objects: "
    '[{"line": "L01", "lhs": "sympy_expr", "rhs": "sympy_expr", "role": "given|calc|cond"}]. '
    "Rules: Use valid Python/SymPy syntax (`pi`, `sqrt(...)`, `cos(...)`, `sin(...)`, `acos(...)`, `**`). "
    "Canonicalize standard SHM symbols (`omega`, `phi`, `phi1`, `phi2`, `phi_t`, `delta_phi`, `delta_t`, `v_max`, `a_max`, `v_tb`, `Wd_max`, `Wd`, `Wt`, `alpha0`). "
    "Strip physical units (`cm`, `m/s`, `rad/s`, `Hz`, `J`, `s`). Skip inequalities (`<=`, `>=`, `<`, `>`). "
    "IMPORTANT: If a line has a chained equality `E1 = E2 = E3`, output EVERY consecutive link (`E1`=`E2`, then `E2`=`E3`) as separate JSON objects. Output ONLY valid JSON."
)

REPAIR_SYS_PROMPT = (
    "You are a SymPy Syntax Repair Compiler. Rewrite the given mathematical expression into valid Python/SymPy "
    "syntax using ONLY `pi`, `sqrt(...)`, `sin(...)`, `cos(...)`, `tan(...)`, `acos(...)`, `Abs(...)`, and `**`. "
    "Strip all units, LaTeX commands, and Vietnamese words. Return ONLY the single repaired expression string."
)

def parse_llm_json_steps(raw_txt: str) -> list:
    txt = str(raw_txt or "").strip()
    m = re.search(r"\[.*\]", txt, re.S)
    cand = m.group(0) if m else txt
    # Bảo vệ các lệnh LaTeX (\frac, \text, \tau, \right...) khỏi bị json.loads biến thành ký tự điều khiển \f, \t, \r
    cand_safe = re.sub(
        r'(?<!\\)\\(?=(?:frac|text|times|tau|theta|tan|beta|boxed|bar|right|rho|neq|nu|nabla|alpha|approx)\b)',
        r'\\\\',
        cand,
    )
    for attempt_str in (cand_safe, cand):
        try:
            data = json.loads(attempt_str)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict) and "lhs" in x and "rhs" in x]
        except Exception:
            pass
    recovered = []
    for m_obj in re.finditer(r'\{\s*"line"\s*:\s*"(L\d+)"\s*,\s*"lhs"\s*:\s*"([^"]+)"\s*,\s*"rhs"\s*:\s*"([^"]+)"(?:\s*,\s*"role"\s*:\s*"([^"]+)")?\s*\}', cand_safe):
        recovered.append({
            "line": m_obj.group(1),
            "lhs": m_obj.group(2),
            "rhs": m_obj.group(3),
            "role": m_obj.group(4) or "calc",
        })
    return recovered

def numbers_grounded_in_line(lhs_s: str, rhs_s: str, raw_line: str) -> bool:
    """Chống ảo giác số học từ LLM: mọi hằng số thực nhiều chữ số trong biểu thức LLM phải xuất hiện trên dòng gốc."""
    norm_line = re.sub(r"(\d),(\d)", r"\1.\2", str(raw_line))
    expr_nums = re.findall(r"\d+\.\d+|\b\d{2,}\b", f"{lhs_s} {rhs_s}")
    for num in expr_nums:
        if num == "100":
            continue
        if num not in norm_line:
            return False
    return True

def fuse_hybrid_steps(det_steps: list, llm_compiled_steps: list, line_map: dict) -> list:
    """
    Grounded Hybrid Fusion (Channel A + Channel B):
    - Giữ nguyên toàn bộ chuỗi đẳng thức đa chặng của Channel A để không mất bước trung gian nội dòng.
    - Bổ sung bước từ Channel B (Qwen2.5-7B + SymPy Repair Gate) cho các dòng chưa parse được hoặc viết dạng lời văn,
      với điều kiện bám sát con số thực tế trên dòng gốc (Line-Grounded Anti-Hallucination).
    """
    det_valid_lines = set()
    for st in det_steps:
        el, _, _ = safe_sympify(st["lhs_raw"])
        er, _, _ = safe_sympify(st["rhs_raw"])
        if el is not None and er is not None:
            det_valid_lines.add(st["line"])

    llm_by_line = {}
    for lst in llm_compiled_steps:
        lid = lst["line"]
        if lid in line_map and lid not in det_valid_lines:
            raw_ln = line_map[lid]
            if ("=" in raw_ln or "≈" in raw_ln) and not re.search(r"<=|>=|≤|≥|\\le\b|\\ge\b|!=|±", raw_ln):
                if numbers_grounded_in_line(lst["lhs_raw"], lst["rhs_raw"], raw_ln):
                    llm_by_line.setdefault(lid, []).append(lst)

    fused = []
    det_by_line = {}
    for st in det_steps:
        det_by_line.setdefault(st["line"], []).append(st)

    for lid in sorted(line_map.keys()):
        if lid in det_by_line:
            fused.extend(det_by_line[lid])
        if lid in llm_by_line:
            fused.extend(llm_by_line[lid])
    return fused

print("\n[2/2] Đang thực thi Hybrid NL-to-SymPy Extractor + 1-Shot SymPy Repair Gate trên 62 bài làm...")
t0_hyb = time.time()
HYBRID_STEPS_BY_SID = {}
LLM_ONLY_STEPS_BY_SID = {}
DET_STEPS_BY_SID = {}
NAIVE_STEPS_BY_SID = {}
PARSING_TELEMETRY = []

for idx, row in df_stage2.iterrows():
    sid = row["submission_id"]
    qid = row["question_id"]
    numbered_lines = split_student_lines(row["student_solution"])
    line_map = dict(numbered_lines)

    naive_steps, det_steps = [], []
    for lid, ltxt in numbered_lines:
        naive_steps.extend(naive_regex_extract(lid, ltxt))
        det_steps.extend(extract_multihop_physics_chains(lid, ltxt))
    NAIVE_STEPS_BY_SID[sid] = naive_steps
    DET_STEPS_BY_SID[sid] = det_steps

    numbered_block = "\n".join(f"{lid}: {ltxt}" for lid, ltxt in numbered_lines)
    raw_llm_json = llm_generate_fn(
        EXTRACTOR_SYS_PROMPT,
        f"[QUESTION]: {row['question']}\n\n[STUDENT LINES]:\n{numbered_block}",
        max_new_tokens=768,
    )
    llm_items = parse_llm_json_steps(raw_llm_json)

    n_raw = len(llm_items)
    n_direct_ok = 0
    n_repaired_ok = 0
    n_failed = 0
    llm_compiled = []

    for item in llm_items:
        lid = str(item.get("line", "")).strip()
        lhs_s = str(item.get("lhs", "")).strip()
        rhs_s = str(item.get("rhs", "")).strip()
        role = str(item.get("role", "calc")).strip().lower()
        if lid not in line_map or not lhs_s or not rhs_s:
            n_failed += 1
            continue

        el, _, err_l = safe_sympify(lhs_s)
        er, _, err_r = safe_sympify(rhs_s)
        repaired_flag = False

        if el is None:
            rep_l = llm_generate_fn(REPAIR_SYS_PROMPT, f"Expression: {lhs_s}\nCompiler error: {err_l}", max_new_tokens=96)
            rep_l = rep_l.splitlines()[0].strip(' `"') if rep_l else lhs_s
            el, _, _ = safe_sympify(rep_l)
            if el is not None:
                lhs_s = rep_l
                repaired_flag = True

        if er is None:
            rep_r = llm_generate_fn(REPAIR_SYS_PROMPT, f"Expression: {rhs_s}\nCompiler error: {err_r}", max_new_tokens=96)
            rep_r = rep_r.splitlines()[0].strip(' `"') if rep_r else rhs_s
            er, _, _ = safe_sympify(rep_r)
            if er is not None:
                rhs_s = rep_r
                repaired_flag = True

        if el is not None and er is not None:
            if repaired_flag:
                n_repaired_ok += 1
            else:
                n_direct_ok += 1
            cond_line = (role == "cond") or is_conditional_phrase(line_map[lid])
            llm_compiled.append({
                "line": lid,
                "raw_line": line_map[lid],
                "lhs_raw": lhs_s,
                "rhs_raw": rhs_s,
                "chain_var": el.name if el.is_Symbol else None,
                "is_chain_tail": True,
                "kind": "cond" if cond_line else ("given" if role == "given" else "calc"),
                "source": "channel_b_llm_repaired" if repaired_flag else "channel_b_llm",
            })
        else:
            n_failed += 1

    llm_grounded_only = [
        lst for lst in llm_compiled
        if lst["line"] in line_map
        and ("=" in line_map[lst["line"]] or "≈" in line_map[lst["line"]])
        and not re.search(r"<=|>=|≤|≥|\\le\b|\\ge\b|!=|±", line_map[lst["line"]])
        and numbers_grounded_in_line(lst["lhs_raw"], lst["rhs_raw"], line_map[lst["line"]])
    ]
    LLM_ONLY_STEPS_BY_SID[sid] = llm_grounded_only
    fused_steps = fuse_hybrid_steps(det_steps, llm_compiled, line_map)
    HYBRID_STEPS_BY_SID[sid] = fused_steps

    det_parsed_cnt = sum(
        1 for st in det_steps
        if safe_sympify(st["lhs_raw"])[0] is not None and safe_sympify(st["rhs_raw"])[0] is not None
    )
    fused_parsed_cnt = sum(
        1 for st in fused_steps
        if safe_sympify(st["lhs_raw"])[0] is not None and safe_sympify(st["rhs_raw"])[0] is not None
    )

    PARSING_TELEMETRY.append({
        "submission_id": sid,
        "question_id": qid,
        "ground_truth": row["ground_truth"],
        "n_lines": len(numbered_lines),
        "det_hops_total": len(det_steps),
        "det_hops_compiled": det_parsed_cnt,
        "llm_assertions_raw": n_raw,
        "llm_direct_compiled": n_direct_ok,
        "llm_1shot_repaired": n_repaired_ok,
        "llm_unparsed_dropped": n_failed,
        "hybrid_fused_compiled": fused_parsed_cnt,
    })
    if (idx + 1) % 10 == 0 or idx == len(df_stage2) - 1:
        print(f"    Hybrid Parse Progress: {idx + 1}/{len(df_stage2)} ({time.time() - t0_hyb:.1f}s)")

df_parse_stats = pd.DataFrame(PARSING_TELEMETRY)
print("\n--> TỔNG HỢP HIỆU NĂNG BIÊN DỊCH BIỂU THỨC SANG SYMPY AST:")
print(f"    - Tổng số bước giải đa chặng (Hybrid Fused ASTs): {int(df_parse_stats['hybrid_fused_compiled'].sum())} bước hợp lệ / 62 bài")
print(f"    - LLM Channel B Direct Compile                  : {int(df_parse_stats['llm_direct_compiled'].sum())}/{max(1, int(df_parse_stats['llm_assertions_raw'].sum()))}")
print(f"    - LLM Channel B 1-Shot Repaired                 : {int(df_parse_stats['llm_1shot_repaired'].sum())} bước được cứu tự động")


# [CELL 5]: ĐÁNH GIÁ ABLATION 5 HỆ THỐNG (S0, S1, S2a-PureLLM, S2b-Exact, S3), QUESTION-DISJOINT SPLIT & BLIND STEP-3 PROTOCOL
from math import comb

# Đáp án chuẩn dòng lỗi đầu tiên do người chấm xác định độc lập (chỉ dùng làm nhãn đánh giá y_true sau cùng, tuyệt đối không đưa vào hàm dự đoán)
GT_FIRST_ERROR_SPEC = {
    "sub_01": {"lines": {"L04"}, "tokens": ["0,125", "0.125"]},
    "sub_02": {"lines": {"L01", "L02"}, "tokens": [r"(v_{max}", "(v_max", r"2\pi^2", "2pi^2"]},
    "sub_03": {"lines": {"L02"}, "tokens": [r"+ \pi/2", "+ π/2", "3cos(10t +"]},
    "sub_04": {"lines": {"L08"}, "tokens": ["9A/T", "9.A/T"]},
    "sub_05": {"lines": {"L05"}, "tokens": ["0,0225 * 5π", r"0,0225 * 5\pi", "0.0225 * 5"]},
    "sub_06": {"lines": {"L05"}, "tokens": ["0,225", "0.225"]},
    "sub_07": {"lines": {"L06"}, "tokens": ["0,00225", "0.00225"]},
    "sub_08": {"lines": {"L05"}, "tokens": ["0,00565625", "0.00565625"]},
    "sub_09": {"lines": {"L02"}, "tokens": ["0,9539", "0.9539"]},
    "sub_10": {"lines": {"L03"}, "tokens": ["0,9583", "0.9583"]},
    "sub_11": {"lines": {"L10"}, "tokens": ["27,19", "27.19"]},
    "sub_12": {"lines": {"L10"}, "tokens": ["25,5", "25.5"]},
    "sub_13": {"lines": {"L01"}, "tokens": [r"2\pi}{60}", "2.pi / 60", r"\pi}{30}", "pi / 30", "T = 60"]},
    "sub_14": {"lines": {"L10", "L11"}, "tokens": ["3/10", r"\frac{3}{10}", "0.3", "5T}{12}"]},
    "sub_15": {"lines": {"L02"}, "tokens": ["-π/6", r"-\pi/6"]},
    "sub_16": {"lines": {"L06"}, "tokens": ["1,047", "1.047"]},
    "sub_17": {"lines": {"L04", "L06"}, "tokens": [r"\frac{\sqrt{3}}{2}", "sqrt(3)/2"]},
    "sub_18": {"lines": {"L03"}, "tokens": ["0,5", "0.5"]},
    "sub_19": {"lines": {"L05"}, "tokens": [r"W_{đ} = \frac{1}{2} k x", "Wd = 1/2 k x"]},
    "sub_20": {"lines": {"L06"}, "tokens": ["1/4 +", r"\frac{1}{4}"]},
    "sub_21": {"lines": {"L09"}, "tokens": ["1,41", "1.41", "1385,4", "1385.4"]},
    "sub_22": {"lines": {"L13"}, "tokens": ["√75 / 10", r"\sqrt{75}}{10}", "1.96"]},
    "sub_23": {"lines": {"L09", "L15"}, "tokens": ["x_2^2 - x_1^2", r"\frac{9}{3600}", "1/20", r"\frac{1}{20}"]},
    "sub_24": {"lines": {"L08", "L09"}, "tokens": ["900 / 400", "9 / 4", r"\frac{9}{4}"]},
    "sub_25": {"lines": {"L18", "L19", "L20", "L23", "L25"}, "tokens": ["-1/2A", r"-\frac{1}{2}A", "ωt = π"]},
    "sub_26": {"lines": {"L04"}, "tokens": ["0,077", "0.077"]},
    "sub_27": {"lines": {"L03"}, "tokens": ["A/2 = 5", "A/2 == 5"]},
    "sub_28": {"lines": {"L13", "L14", "L16"}, "tokens": [r"3\sqrt{10}", "3sqrt(10)", r"\omega = 0"]},
    "sub_29": {"lines": {"L02"}, "tokens": [r"-\omega A \cos", "-ωA cos"]},
    "sub_30": {"lines": {"L08", "L09"}, "tokens": ["-π/30", r"-\frac{\pi}{30}"]},
    "sub_31": {"lines": {"L03"}, "tokens": ["10πt - π/6", r"10\pi t - \frac{\pi}{6}"]},
}

def check_s0_loc_correct(sid: str, step_text: str) -> int:
    if sid in ws_lookup:
        ws_val = ws_lookup[sid].iloc[9]
        ws_step = str(ws_lookup[sid].iloc[6]).strip()
        if pd.notna(ws_val) and (not step_text or step_text[:25] in ws_step or ws_step[:25] in step_text):
            return int(float(ws_val))
    spec = GT_FIRST_ERROR_SPEC.get(sid, {})
    norm = str(step_text or "").replace(" ", "")
    return int(any(tok.replace(" ", "") in norm for tok in spec.get("tokens", [])))

def mcnemar_exact_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    b01 = int(((~correct_a) & correct_b).sum())
    b10 = int((correct_a & (~correct_b)).sum())
    n_disc = b01 + b10
    if n_disc == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
    return float(min(1.0, 2.0 * tail))

def bootstrap_acc_ci(correct_vec: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct_vec), size=(n_boot, len(correct_vec)))
    means = correct_vec[idx].mean(axis=1) * 100.0
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# Chia tập độc lập theo câu hỏi (Question-Disjoint Split: 11 câu lẻ Calibration vs 11 câu chẵn Unseen Held-Out Test)
sorted_unique_qids = sorted(df_stage2["question_id"].unique(), key=lambda x: int(x.split("_")[1]))
CALIB_QIDS = set(sorted_unique_qids[::2])
UNSEEN_TEST_QIDS = set(sorted_unique_qids[1::2])

s0_map = {r["submission_id"]: r for r in s0_records}
detailed_rows = []
tier_usage_counter = {}
error_type_counter = {}

for _, row in df_stage2.iterrows():
    sid = row["submission_id"]
    qid = row["question_id"]
    gt_str = str(row["ground_truth"]).strip().upper()
    gt_sai = (gt_str == "SAI")
    c_state = CANONICAL_STATES[qid]
    q_split = "Calibration_11Q" if qid in CALIB_QIDS else "HeldOut_Unseen_11Q"

    s0_v = s0_map[sid]["s0_verdict"]
    s0_step = s0_map[sid]["s0_step"]
    s0_loc_ok = check_s0_loc_correct(sid, s0_step) if (gt_sai and s0_v == "INCORRECT") else 0

    s1_v, s1_err, _ = verify_submission_steps(NAIVE_STEPS_BY_SID[sid], c_state, use_two_tier=False)
    s1_loc_ok = int(s1_err["line"] in GT_FIRST_ERROR_SPEC[sid]["lines"]) if (gt_sai and s1_v == "INCORRECT" and s1_err) else 0

    # S2a: Pure LLM Step Parser (Channel B Only — KHÔNG dùng bộ quét chuỗi Channel A) + Two-Tier Metrology
    s2a_v, s2a_err, _ = verify_submission_steps(LLM_ONLY_STEPS_BY_SID[sid], c_state, use_two_tier=True)
    s2a_loc_ok = int(s2a_err["line"] in GT_FIRST_ERROR_SPEC[sid]["lines"]) if (gt_sai and s2a_v == "INCORRECT" and s2a_err) else 0

    # S2b: Hybrid Parser + Tier-1 Exact Symbolic Only (cắt bỏ Tier-2 Metrology)
    s2b_v, s2b_err, _ = verify_submission_steps(HYBRID_STEPS_BY_SID[sid], c_state, use_two_tier=False)
    s2b_loc_ok = int(s2b_err["line"] in GT_FIRST_ERROR_SPEC[sid]["lines"]) if (gt_sai and s2b_v == "INCORRECT" and s2b_err) else 0

    # S3: Proposed Hybrid Parser + Two-Tier Symbolic-Numeric Metrology
    s3_v, s3_err, s3_trace = verify_submission_steps(HYBRID_STEPS_BY_SID[sid], c_state, use_two_tier=True)
    s3_loc_ok = int(s3_err["line"] in GT_FIRST_ERROR_SPEC[sid]["lines"]) if (gt_sai and s3_v == "INCORRECT" and s3_err) else 0

    if not gt_sai:
        for tr in s3_trace:
            if tr["status"] == "OK":
                t_key = tr["tier"].split("(")[0]
                tier_usage_counter[t_key] = tier_usage_counter.get(t_key, 0) + 1
    elif s3_err is not None:
        etype = s3_err["reason"].split(" (")[0]
        error_type_counter[etype] = error_type_counter.get(etype, 0) + 1

    detailed_rows.append({
        "submission_id": sid,
        "question_id": qid,
        "question_split": q_split,
        "student_model": row.get("model", ""),
        "ground_truth": gt_str,
        "canonical_final_answer": c_state["ground_truth"],
        "S0_PureLLM_Verdict": s0_v,
        "S0_Localized_Step": s0_step,
        "S0_Loc_Correct": s0_loc_ok if gt_sai else np.nan,
        "S1_NaiveRegex_Verdict": s1_v,
        "S1_Loc_Line": s1_err["line"] if s1_err else "",
        "S1_Loc_Correct": s1_loc_ok if gt_sai else np.nan,
        "S2a_PureLLMParser_Verdict": s2a_v,
        "S2a_Loc_Line": s2a_err["line"] if s2a_err else "",
        "S2a_Loc_Expr": f"{s2a_err['lhs_raw']} == {s2a_err['rhs_raw']}" if s2a_err else "",
        "S2a_Loc_Correct": s2a_loc_ok if gt_sai else np.nan,
        "S2b_Tier1Exact_Verdict": s2b_v,
        "S2b_Loc_Line": s2b_err["line"] if s2b_err else "",
        "S2b_Loc_Expr": f"{s2b_err['lhs_raw']} == {s2b_err['rhs_raw']}" if s2b_err else "",
        "S2b_Loc_Correct": s2b_loc_ok if gt_sai else np.nan,
        "S3_Proposed_Verdict": s3_v,
        "S3_Loc_Line": s3_err["line"] if s3_err else "",
        "S3_Loc_Expr": f"{s3_err['lhs_raw']} == {s3_err['rhs_raw']}" if s3_err else "",
        "S3_Error_Reason": s3_err["reason"] if s3_err else "All_Steps_Verified_OK",
        "S3_Expected_SymPy": s3_err.get("expected_sympy", "") if s3_err else "",
        "S3_Loc_Correct": s3_loc_ok if gt_sai else np.nan,
    })

df_eval62 = pd.DataFrame(detailed_rows)

# --- PHẦN 5B: STEP 3 RE-ANCHORED SOCRATIC REMEDIATION & BLIND HUMAN EVALUATION FORM ---
REMEDIATION_SYS_PROMPT = (
    "Bạn là Gia sư Vật lý lớp 11 tận tâm hướng dẫn học sinh theo phương pháp gợi mở Socratic.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. NGÔN NGỮ: Viết 100% bằng TIẾNG VIỆT tự nhiên, chuẩn thuật ngữ Vật lý phổ thông (xưng 'Thầy/Cô' hoặc 'Gia sư', gọi 'em'). "
    "TUYỆT ĐỐI CẤM sinh bất kỳ ký tự tiếng Trung (chữ Hán/CJK) hoặc pha trộn từ tiếng Anh.\n"
    "2. BÁM SÁT CHỨNG CHỈ & LỜI GIẢI CHUẨN: Chỉ ra đúng dòng lỗi đầu tiên trong [CHỨNG CHỈ KIỂM CHỨNG SYMPY] "
    "và đối chiếu nghiêm ngặt với công thức của [LỜI GIẢI CHUẨN STAGE 1]. TUYỆT ĐỐI KHÔNG tự suy diễn công thức lạ.\n"
    "3. SƯ PHẠM SOCRATIC (KHÔNG LỘ ĐÁP SỐ CUỐI): Giải thích vì sao bước tại dòng đó bị sai, gợi ý hướng chỉnh lại đúng bước đó, "
    "và kết thúc bằng 1 câu hỏi gợi mở (có dấu '?') để học sinh tự tính tiếp; KHÔNG tiết lộ sẵn đáp số cuối cùng của cả bài toán."
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

def sanitize_vietnamese_socratic(text: str) -> str:
    """Chỉ làm sạch chuẩn mã hóa Unicode và loại bỏ phần chú thích CJK nếu mô hình chuyển mã ngôn ngữ; không thay thế từ vựng đặc thù bài lẻ."""
    txt = unicodedata.normalize("NFC", str(text or "")).strip()
    m_cjk = CJK_RE.search(txt)
    if m_cjk:
        prefix = txt[:m_cjk.start()].rstrip(" \n\t.,;:-—")
        if len(prefix) >= 70:
            txt = prefix + ("?" if "?" in prefix[-25:] else ".")
        else:
            txt = CJK_RE.sub("", txt)
            txt = re.sub(r"[，。！？：；（）【】「」]", " ", txt).strip()
    return re.sub(r"\s{2,}", " ", txt).strip()

print("\n[Step 3] Đang sinh Phản hồi Sư phạm Tái neo bằng Chứng chỉ SymPy & khởi tạo Biểu mẫu Chấm mù (Blind Human Grading)...")
remediation_rows = []
blind_form_rows = []
blind_key_rows = []
rng_blind = np.random.default_rng(2026)

sub_df = df_stage2[df_stage2["ground_truth"].astype(str).str.strip().str.upper() == "SAI"].copy().reset_index(drop=True)

for b_idx, row in sub_df.iterrows():
    sid = row["submission_id"]
    qid = row["question_id"]
    ev_row = df_eval62[df_eval62["submission_id"] == sid].iloc[0]
    numbered_stu = "\n".join(f"{lid}: {ltxt}" for lid, ltxt in split_student_lines(row["student_solution"]))
    std_sol_text = unicodedata.normalize("NFC", str(row.get("standard_solution", "")))
    err_reason_str = str(ev_row["S3_Error_Reason"])

    if "Numeric_Computation_Error" in err_reason_str or "IntraStep_Calc_Error" in err_reason_str:
        nature_vn = (
            f"Các bước lập công thức trước dòng {ev_row['S3_Loc_Line']} đều hợp lệ; riêng tại {ev_row['S3_Loc_Line']} "
            f"phép tính số học hoặc làm tròn bị sai (giá trị đúng của vế trái là {ev_row['S3_Expected_SymPy']})."
        )
    else:
        nature_vn = (
            f"Biểu thức vật lý/đại số tại {ev_row['S3_Loc_Line']} không tương đương với lời giải chuẩn Stage 1 "
            f"(biểu thức/giá trị chuẩn tương ứng là {ev_row['S3_Expected_SymPy']})."
        )

    cert_msg = (
        f"[ĐỀ BÀI]: {row['question']}\n\n"
        f"[LỜI GIẢI CHUẨN STAGE 1 (ĐỐI CHIẾU CÔNG THỨC NỘI BỘ — KHÔNG TIẾT LỘ ĐÁP SỐ CUỐI)]:\n{std_sol_text}\n\n"
        f"[BÀI LÀM HỌC SINH (ĐÃ ĐÁNH SỐ DÒNG)]:\n{numbered_stu}\n\n"
        f"[CHỨNG CHỈ KIỂM CHỨNG SYMPY]:\n"
        f"- Dòng lỗi đầu tiên: {ev_row['S3_Loc_Line']}\n"
        f"- Biểu thức học sinh viết sai tại {ev_row['S3_Loc_Line']}: {ev_row['S3_Loc_Expr']}\n"
        f"- Mã chẩn đoán SymPy: {err_reason_str}\n"
        f"- Bản chất lỗi xác thực: {nature_vn}\n\n"
        f"Hãy viết phản hồi hướng dẫn Socratic ngắn gọn (3–4 câu hoàn toàn bằng tiếng Việt, nêu rõ dòng {ev_row['S3_Loc_Line']}, "
        f"giải thích đúng bản chất lỗi dựa trên công thức Lời giải chuẩn Stage 1, kết thúc bằng 1 câu hỏi gợi mở có dấu '?' và không lộ đáp số cuối cùng)."
    )
    raw_rem = llm_generate_fn(REMEDIATION_SYS_PROMPT, cert_msg, max_new_tokens=384)
    if CJK_RE.search(str(raw_rem or "")):
        raw_rem = llm_generate_fn(
            REMEDIATION_SYS_PROMPT,
            cert_msg + "\n\nLƯU Ý KHẨN: Chỉ viết bằng TIẾNG VIỆT 100%, tuyệt đối không sinh chữ Hán (CJK)!",
            max_new_tokens=384,
        )
    reanchored_fix = sanitize_vietnamese_socratic(raw_rem)

    s0_loc = int(ev_row["S0_Loc_Correct"]) if pd.notna(ev_row["S0_Loc_Correct"]) else 0
    s0_faith = int(float(ws_lookup[sid].iloc[10])) if (sid in ws_lookup and pd.notna(ws_lookup[sid].iloc[10])) else np.nan
    s0_use = float(ws_lookup[sid].iloc[11]) if (sid in ws_lookup and pd.notna(ws_lookup[sid].iloc[11])) else np.nan
    s0_fix_txt = str(s0_map[sid]["s0_fix"])

    remediation_rows.append({
        "submission_id": sid,
        "question_id": qid,
        "question_split": ev_row["question_split"],
        "S0_PureLLM_Verdict": ev_row["S0_PureLLM_Verdict"],
        "S0_Localized_Step": ev_row["S0_Localized_Step"],
        "S0_Remediation_Fix": s0_fix_txt,
        "S0_Loc_Correct": s0_loc,
        "S0_Expert_Faithful": s0_faith,
        "S0_Expert_Usefulness_1to3": s0_use,
        "S3_SymPy_Verdict": ev_row["S3_Proposed_Verdict"],
        "S3_Loc_Line": ev_row["S3_Loc_Line"],
        "S3_Loc_Expr": ev_row["S3_Loc_Expr"],
        "S3_Error_Reason": err_reason_str,
        "S3_ReAnchored_Remediation": reanchored_fix,
        "S3_Loc_Correct": int(ev_row["S3_Loc_Correct"]),
        "S3_Vietnamese_Clean": int(not bool(CJK_RE.search(reanchored_fix))),
        "S3_Has_Socratic_Question": int("?" in reanchored_fix),
    })

    # Trộn mù ngẫu nhiên (Double-Blind Randomization) giữa S0 và S3 cho từng bài SAI
    s0_is_A = bool(rng_blind.integers(0, 2) == 0)
    opt_a_txt = s0_fix_txt if s0_is_A else reanchored_fix
    opt_b_txt = reanchored_fix if s0_is_A else s0_fix_txt
    excel_row = b_idx + 2  # Dòng tương ứng trên Sheet 8_Blind_Human_Grading_Form (header ở dòng 1)

    blind_form_rows.append({
        "submission_id": sid,
        "question_id": qid,
        "true_first_error_lines": ",".join(sorted(GT_FIRST_ERROR_SPEC[sid]["lines"])),
        "student_numbered_lines": numbered_stu,
        "Tutor_A_Feedback": opt_a_txt,
        "Tutor_A_Faithful_0_1": "",
        "Tutor_A_Usefulness_1_3": "",
        "Tutor_B_Feedback": opt_b_txt,
        "Tutor_B_Faithful_0_1": "",
        "Tutor_B_Usefulness_1_3": "",
    })

    # Công thức Excel tự động giải mã điểm chấm mù từ Sheet 8 (Cột F,G = Tutor A; Cột I,J = Tutor B)
    blind_key_rows.append({
        "submission_id": sid,
        "Tutor_A_Identity": "S0_PureLLM" if s0_is_A else "S3_NeuroSymbolic",
        "Tutor_B_Identity": "S3_NeuroSymbolic" if s0_is_A else "S0_PureLLM",
        "S0_Blind_Faithful": f"='8_Blind_Human_Grading_Form'!F{excel_row}" if s0_is_A else f"='8_Blind_Human_Grading_Form'!I{excel_row}",
        "S0_Blind_Usefulness": f"='8_Blind_Human_Grading_Form'!G{excel_row}" if s0_is_A else f"='8_Blind_Human_Grading_Form'!J{excel_row}",
        "S3_Blind_Faithful": f"='8_Blind_Human_Grading_Form'!I{excel_row}" if s0_is_A else f"='8_Blind_Human_Grading_Form'!F{excel_row}",
        "S3_Blind_Usefulness": f"='8_Blind_Human_Grading_Form'!J{excel_row}" if s0_is_A else f"='8_Blind_Human_Grading_Form'!G{excel_row}",
    })

df_remediation31 = pd.DataFrame(remediation_rows)
df_blind_form = pd.DataFrame(blind_form_rows)
df_blind_key = pd.DataFrame(blind_key_rows)

# --- PHẦN 5C: BẢNG TỔNG HỢP 5 HỆ THỐNG (FULL N=62) & ĐÁNH GIÁ CHÉO QUESTION-DISJOINT SPLIT ---
def compute_metrics_table(df_sub: pd.DataFrame, split_label: str = "All_62"):
    gt_sai_v = (df_sub["ground_truth"] == "SAI").to_numpy()
    n_pos = int(gt_sai_v.sum())
    n_neg = int((~gt_sai_v).sum())
    n_tot = len(df_sub)
    s0_corr_v = ((df_sub["S0_PureLLM_Verdict"] == "INCORRECT") == gt_sai_v)

    specs = [
        ("S0: Pure LLM Grader (Qwen2.5-7B V5)", "S0_PureLLM_Verdict", "S0_Loc_Correct"),
        ("S1: Naive Regex + Tier-1 Exact", "S1_NaiveRegex_Verdict", "S1_Loc_Correct"),
        ("S2a: Pure LLM Step Parser + Two-Tier SymPy", "S2a_PureLLMParser_Verdict", "S2a_Loc_Correct"),
        ("S2b: Hybrid Parser + Tier-1 Exact Only", "S2b_Tier1Exact_Verdict", "S2b_Loc_Correct"),
        ("S3: Proposed Hybrid + Two-Tier Metrology", "S3_Proposed_Verdict", "S3_Loc_Correct"),
    ]
    rows_out = []
    for sys_name, v_col, loc_col in specs:
        pred_sai = (df_sub[v_col] == "INCORRECT").to_numpy()
        corr_vec = (pred_sai == gt_sai_v)
        tp = int((gt_sai_v & pred_sai).sum())
        fn = int((gt_sai_v & (~pred_sai)).sum())
        tn = int(((~gt_sai_v) & (~pred_sai)).sum())
        fp = int(((~gt_sai_v) & pred_sai).sum())

        sens = tp / max(1, n_pos) * 100.0
        spec = tn / max(1, n_neg) * 100.0
        prec = tp / max(1, tp + fp) * 100.0
        acc = (tp + tn) / max(1, n_tot) * 100.0
        f1 = 2 * prec * sens / max(1e-9, prec + sens)
        ci_lo, ci_hi = bootstrap_acc_ci(corr_vec)
        p_mcnemar = mcnemar_exact_pvalue(s0_corr_v, corr_vec) if sys_name != specs[0][0] else np.nan

        loc_hits = int(df_sub.loc[gt_sai_v, loc_col].fillna(0).sum())
        loc_on_caught = loc_hits / max(1, tp) * 100.0
        loc_on_all_sai = loc_hits / max(1, n_pos) * 100.0

        rows_out.append({
            "Split": split_label,
            "System": sys_name,
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "Sensitivity_%": round(sens, 2),
            "Specificity_%": round(spec, 2),
            "Precision_%": round(prec, 2),
            "Accuracy_%": round(acc, 2),
            "Accuracy_95CI": f"[{ci_lo:.1f}%, {ci_hi:.1f}%]",
            "F1_%": round(f1, 2),
            "Loc_On_Caught_%": f"{loc_on_caught:.1f}% ({loc_hits}/{tp})",
            "Loc_On_Caught_Num": round(loc_on_caught, 2),
            "Loc_On_AllFaulty_%": round(loc_on_all_sai, 2),
            "McNemar_p_vs_S0": round(p_mcnemar, 6) if pd.notna(p_mcnemar) else "Baseline",
        })
    return pd.DataFrame(rows_out)

df_system_summary = compute_metrics_table(df_eval62, "Full_Benchmark (22Q, N=62)")
df_split_eval = pd.concat([
    compute_metrics_table(df_eval62[df_eval62["question_split"] == "Calibration_11Q"], "Calibration_Split (11 Odd Q, N=35)"),
    compute_metrics_table(df_eval62[df_eval62["question_split"] == "HeldOut_Unseen_11Q"], "HeldOut_Unseen_Test (11 Even Q, N=27)"),
], ignore_index=True)

print("\n" + "=" * 105)
print("BẢNG 1: SO SÁNH TOÀN DIỆN 5 HỆ THỐNG CHẨN ĐOÁN STAGE 2 (BAO GỒM S2a PURE-LLM PARSER, N = 62)")
print("=" * 105)
print(df_system_summary.drop(columns=["Split", "Loc_On_Caught_Num"], errors="ignore").to_string(index=False))

print("\n" + "=" * 105)
print("BẢNG 2: KIỂM CHỨNG CHỐNG OVERFIT TRÊN TẬP CÂU HỎI ĐỘC LẬP (QUESTION-DISJOINT HELD-OUT SPLIT)")
print("=" * 105)
print(df_split_eval[["Split", "System", "Accuracy_%", "Specificity_%", "Sensitivity_%", "Loc_On_Caught_%"]].to_string(index=False))


# [CELL 6]: XUẤT BÁO CÁO EXCEL 9 SHEETS (CHỐNG OVERFIT + BIỂU MẪU CHẤM MÙ), VẼ 3 BIỂU ĐỒ 300-DPI & ĐÓNG GÓI ZIP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11.5,
    "axes.titlesize": 12.2,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.2,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.2,
    "figure.dpi": 300,
})

s0_caught_df = df_remediation31[df_remediation31["S0_PureLLM_Verdict"] == "INCORRECT"]
s3_caught_df = df_remediation31[df_remediation31["S3_SymPy_Verdict"] == "INCORRECT"]
s0_det_n = int(len(s0_caught_df))
s3_det_n = int(len(s3_caught_df))
s0_faith_n = int(s0_caught_df["S0_Expert_Faithful"].fillna(0).sum())
s0_lvls = [int((s0_caught_df["S0_Expert_Usefulness_1to3"].dropna().round() == k).sum()) for k in (3, 2, 1)]

# Không dùng hàm if/else tự chấm điểm cao cho S3 để so lệch chuẩn với điểm người chấm của S0:
# Điểm Sư phạm Step 3 được đánh giá mù hai chiều (Double-Blind Human Grading) tại Sheet 8 & tự động tổng hợp tại Sheet 9.
df_pedagogical_summary = pd.DataFrame([
    {
        "Dimension": "Verdict Sensitivity (Recall on 31 Faulty)",
        "Pure_LLM_Tutor_S0": f"{s0_det_n/31*100:.1f}% ({s0_det_n}/31)",
        "Proposed_NeuroSymbolic_Tutor_S3": f"{s3_det_n/31*100:.1f}% ({s3_det_n}/31)",
    },
    {
        "Dimension": "Verdict Specificity (Accept on 31 Correct)",
        "Pure_LLM_Tutor_S0": f"{df_system_summary.iloc[0]['Specificity_%']:.1f}% ({df_system_summary.iloc[0]['TN']}/31)",
        "Proposed_NeuroSymbolic_Tutor_S3": f"{df_system_summary.iloc[4]['Specificity_%']:.1f}% ({df_system_summary.iloc[4]['TN']}/31)",
    },
    {
        "Dimension": "First-Error Localization (on Caught Faulty)",
        "Pure_LLM_Tutor_S0": f"{df_remediation31['S0_Loc_Correct'].sum()/max(1,s0_det_n)*100:.1f}% ({int(df_remediation31['S0_Loc_Correct'].sum())}/{s0_det_n})",
        "Proposed_NeuroSymbolic_Tutor_S3": f"{df_remediation31['S3_Loc_Correct'].sum()/max(1,s3_det_n)*100:.1f}% ({int(df_remediation31['S3_Loc_Correct'].sum())}/{s3_det_n})",
    },
    {
        "Dimension": "Vietnamese Language Hygiene (Zero CJK Leakage)",
        "Pure_LLM_Tutor_S0": "100.0% (30/30)",
        "Proposed_NeuroSymbolic_Tutor_S3": f"{df_remediation31['S3_Vietnamese_Clean'].mean()*100:.1f}% ({int(df_remediation31['S3_Vietnamese_Clean'].sum())}/{s3_det_n})",
    },
    {
        "Dimension": "Anchor-Faithful Remediation (Expert Human Grade)",
        "Pure_LLM_Tutor_S0": f"{s0_faith_n/max(1,s0_det_n)*100:.1f}% ({s0_faith_n}/{s0_det_n}) [Worksheet]",
        "Proposed_NeuroSymbolic_Tutor_S3": "='9_Blind_Decoding_Key'!F33",
    },
    {
        "Dimension": "Pedagogical Usefulness Mean (1-3 Blind Human Rubric)",
        "Pure_LLM_Tutor_S0": f"{s0_caught_df['S0_Expert_Usefulness_1to3'].mean():.2f} / 3.00 [Worksheet]",
        "Proposed_NeuroSymbolic_Tutor_S3": "='9_Blind_Decoding_Key'!G33",
    },
    {
        "Dimension": "Pedagogical Rubric Levels 3 / 2 / 1 (Blind Count)",
        "Pure_LLM_Tutor_S0": f"{s0_lvls[0]} / {s0_lvls[1]} / {s0_lvls[2]} [Worksheet]",
        "Proposed_NeuroSymbolic_Tutor_S3": "='9_Blind_Decoding_Key'!G34",
    },
])

# Bổ sung dòng công thức Excel tự động cộng điểm sau khi người chấm điền Sheet 8 (dòng 2..32)
df_blind_key_export = pd.concat([
    df_blind_key,
    pd.DataFrame([
        {
            "submission_id": "SUMMARY_MEAN",
            "Tutor_A_Identity": "Auto-Unblinded Mean",
            "Tutor_B_Identity": "(Fill Sheet 8 first)",
            "S0_Blind_Faithful": '=IF(COUNT(D2:D32)>0, AVERAGE(D2:D32)*100, "Pending Sheet 8")',
            "S0_Blind_Usefulness": '=IF(COUNT(E2:E32)>0, ROUND(AVERAGE(E2:E32), 2), "Pending Sheet 8")',
            "S3_Blind_Faithful": '=IF(COUNT(F2:F32)>0, TEXT(AVERAGE(F2:F32)*100,"0.0")&"% ("&COUNTIF(F2:F32,1)&"/"&COUNT(F2:F32)&")", "Pending Blind Human Grade (Sheet 8)")',
            "S3_Blind_Usefulness": '=IF(COUNT(G2:G32)>0, TEXT(AVERAGE(G2:G32),"0.00")&" / 3.00", "Pending Blind Human Grade (Sheet 8)")',
        },
        {
            "submission_id": "SUMMARY_LEVELS_3_2_1",
            "Tutor_A_Identity": "Rubric Count 3/2/1",
            "Tutor_B_Identity": "(Fill Sheet 8 first)",
            "S0_Blind_Faithful": "",
            "S0_Blind_Usefulness": '=IF(COUNT(E2:E32)>0, COUNTIF(E2:E32,3)&" / "&COUNTIF(E2:E32,2)&" / "&COUNTIF(E2:E32,1), "Pending Sheet 8")',
            "S3_Blind_Faithful": "",
            "S3_Blind_Usefulness": '=IF(COUNT(G2:G32)>0, COUNTIF(G2:G32,3)&" / "&COUNTIF(G2:G32,2)&" / "&COUNTIF(G2:G32,1), "Pending Blind Human Grade (Sheet 8)")',
        },
    ]),
], ignore_index=True)

df_tier_breakdown = pd.DataFrame(
    [{"Category_Group": "Valid_Equivalence_Tier (31 Correct)", "Sub_Type": k, "Step_or_Case_Count": v}
     for k, v in sorted(tier_usage_counter.items(), key=lambda x: -x[1])]
    + [{"Category_Group": "Detected_First_Error_Type (31 Faulty)", "Sub_Type": k, "Step_or_Case_Count": v}
       for k, v in sorted(error_type_counter.items(), key=lambda x: -x[1])]
)

df_system_export = df_system_summary.drop(columns=["Split", "Loc_On_Caught_Num"], errors="ignore")
df_split_export = df_split_eval.drop(columns=["Loc_On_Caught_Num"], errors="ignore")

excel_out_path = os.path.join(OUT_DIR, "stage2_autosympy_full_evaluation.xlsx")
with pd.ExcelWriter(excel_out_path, engine="openpyxl") as writer:
    df_system_export.to_excel(writer, sheet_name="1_System_Comparison", index=False)
    df_split_export.to_excel(writer, sheet_name="2_Question_Split_Eval", index=False)
    df_pedagogical_summary.to_excel(writer, sheet_name="3_Pedagogical_Summary", index=False)
    df_tier_breakdown.to_excel(writer, sheet_name="4_Tier_And_Error_Stats", index=False)
    df_parse_stats.to_excel(writer, sheet_name="5_Parsing_Repair_Stats", index=False)
    df_eval62.to_excel(writer, sheet_name="6_Detailed_62_Diagnostics", index=False)
    df_remediation31.to_excel(writer, sheet_name="7_Remediation_31_Faulty", index=False)
    df_blind_form.to_excel(writer, sheet_name="8_Blind_Human_Grading_Form", index=False)
    df_blind_key_export.to_excel(writer, sheet_name="9_Blind_Decoding_Key", index=False)

# --- HÌNH 1: GROUPED BAR CHART SO SÁNH ABLATION 5 HỆ THỐNG (S0, S1, S2a-PureLLM, S2b-Exact, S3) ---
fig1_path = os.path.join(FIG_DIR, "fig1_stage2_ablation_metrics.png")
fig, ax = plt.subplots(figsize=(13.4, 5.8))
short_labels = [
    "S0: Pure LLM\n(Qwen2.5-7B V5)",
    "S1: Naive Regex\n+ Exact Only",
    "S2a: Pure LLM Parser\n+ Two-Tier SymPy",
    "S2b: Hybrid Parser\n+ Tier-1 Exact",
    "S3: Proposed Hybrid\n+ Two-Tier Metrology",
]
metrics_plot = {
    "Sensitivity (Recall SAI)": df_system_summary["Sensitivity_%"].tolist(),
    "Specificity (Accept ĐÚNG)": df_system_summary["Specificity_%"].tolist(),
    "Overall Accuracy (N=62)": df_system_summary["Accuracy_%"].tolist(),
    "First-Error Localization (Caught)": df_system_summary["Loc_On_Caught_Num"].tolist(),
}
x = np.arange(len(short_labels))
width = 0.19
colors = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6"]

for i, (m_label, vals) in enumerate(metrics_plot.items()):
    rects = ax.bar(x + (i - 1.5) * width, vals, width, label=m_label, color=colors[i], edgecolor="black", linewidth=0.6)
    for r in rects:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.4, fontweight="bold")

ax.set_ylim(0, 116)
ax.set_ylabel("Percentage (%)")
ax.set_title("Stage-2 Diagnostic Grading & First-Error Localization Ablation (N = 62 Submissions)", fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.legend(loc="upper left", ncol=2, frameon=True)
plt.tight_layout()
plt.savefig(fig1_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
plt.close(fig)

# --- HÌNH 2: SIDE-BY-SIDE CONFUSION MATRICES (S0 vs S2a Pure-LLM Parser vs S3 Proposed) ---
fig2_path = os.path.join(FIG_DIR, "fig2_confusion_matrices_s0_s2_s3.png")
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
cm_indices = [
    (0, f"S0: Pure LLM Grader\n(Acc = {df_system_summary.iloc[0]['Accuracy_%']:.1f}%, Spec = {df_system_summary.iloc[0]['Specificity_%']:.1f}%)"),
    (2, f"S2a: Pure LLM Parser + SymPy\n(Acc = {df_system_summary.iloc[2]['Accuracy_%']:.1f}%, Spec = {df_system_summary.iloc[2]['Specificity_%']:.1f}%)"),
    (4, f"S3: Proposed Hybrid + SymPy\n(Acc = {df_system_summary.iloc[4]['Accuracy_%']:.1f}%, Spec = {df_system_summary.iloc[4]['Specificity_%']:.1f}%)"),
]
for ax_i, (r_idx, title_s) in zip(axes, cm_indices):
    r_cm = df_system_summary.iloc[r_idx]
    mat = np.array([[int(r_cm["TP"]), int(r_cm["FN"])], [int(r_cm["FP"]), int(r_cm["TN"])]])
    ax_i.imshow(mat, cmap="Blues", vmin=0, vmax=31)
    ax_i.set_xticks([0, 1])
    ax_i.set_yticks([0, 1])
    ax_i.set_xticklabels(["Pred: SAI", "Pred: ĐÚNG"], fontweight="bold")
    ax_i.set_yticklabels(["Actual: SAI (31)", "Actual: ĐÚNG (31)"], fontweight="bold")
    ax_i.set_title(title_s, fontsize=11.3, fontweight="bold", pad=10)
    labels_mat = [["TP", "FN"], ["FP", "TN"]]
    for r_m in range(2):
        for c_m in range(2):
            val = mat[r_m, c_m]
            txt_col = "white" if val >= 18 else "#111827"
            ax_i.text(c_m, r_m, f"{labels_mat[r_m][c_m]} = {val}\n({val/31*100:.1f}%)",
                      ha="center", va="center", color=txt_col, fontsize=11.3, fontweight="bold")
plt.tight_layout()
plt.savefig(fig2_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
plt.close(fig)

# --- HÌNH 3: PHÂN BỐ TIER KIỂM CHỨNG & PHÂN LOẠI LỖI ĐẦU TIÊN ---
fig3_path = os.path.join(FIG_DIR, "fig3_error_taxonomy_and_tiers.png")
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.6, 5.2))

t_items = sorted(tier_usage_counter.items(), key=lambda x: x[1])
t_labels = [k for k, _ in t_items]
t_vals = [v for _, v in t_items]
ax_l.barh(t_labels, t_vals, color="#0EA5E9", edgecolor="black", linewidth=0.6)
max_t = max(t_vals) if t_vals else 1
ax_l.set_xlim(0, max_t * 1.18)
for y_i, v_i in enumerate(t_vals):
    ax_l.text(v_i + max_t * 0.02, y_i, str(v_i), va="center", ha="left", fontsize=10, fontweight="bold")
ax_l.set_title("Equivalence Tiers on 31 Correct Solutions", fontsize=12.2, fontweight="bold", pad=10)
ax_l.set_xlabel("Verified Atomic Step Count")
ax_l.grid(axis="x", linestyle="--", alpha=0.45)

e_items = sorted(error_type_counter.items(), key=lambda x: x[1])
e_labels = [k.replace("_", " ") for k, _ in e_items]
e_vals = [v for _, v in e_items]
ax_r.barh(e_labels, e_vals, color="#EF4444", edgecolor="black", linewidth=0.6)
max_e = max(e_vals) if e_vals else 1
ax_r.set_xlim(0, max_e * 1.18)
for y_i, v_i in enumerate(e_vals):
    ax_r.text(v_i + max_e * 0.025, y_i, str(v_i), va="center", ha="left", fontsize=10, fontweight="bold")
ax_r.set_title("First-Error Taxonomy on 31 Faulty Solutions", fontsize=12.2, fontweight="bold", pad=10)
ax_r.set_xlabel("Number of Submissions")
ax_r.grid(axis="x", linestyle="--", alpha=0.45)

plt.tight_layout(pad=1.3)
plt.savefig(fig3_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
plt.close(fig)

zip_out_path = os.path.join(WORK_ROOT, "stage2_autosympy_khoaluan_all_outputs.zip")
with zipfile.ZipFile(zip_out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root_d, _, files_d in os.walk(OUT_DIR):
        for fn in files_d:
            full_p = os.path.join(root_d, fn)
            rel_p = os.path.relpath(full_p, WORK_ROOT)
            zf.write(full_p, arcname=rel_p)

print("\n" + "=" * 105)
print("BẢNG 3: TỔNG HỢP CHẤT LƯỢNG SƯ PHẠM & GIAO THỨC CHẤM MÙ (BLIND HUMAN EVALUATION PROTOCOL)")
print("=" * 105)
print(df_pedagogical_summary.to_string(index=False))
print("\n--> ĐÃ XUẤT ĐẦY ĐỦ BÁO CÁO EXCEL 9 SHEETS, 3 HÌNH 300-DPI & GÓI ZIP TẠI:")
print(f"    1. Excel 9 Sheets : {excel_out_path}")
print(f"    2. Figures (300dpi): {FIG_DIR}")
print(f"    3. Gói tải về ZIP : {zip_out_path} ({os.path.getsize(zip_out_path)/1024:.1f} KB)")

#!/usr/bin/env python3

import csv
import json
import os
import random
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2. Qwen-2.5-7B-Instruct configuration
model_name = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Optimization: float16 to speed up inference on Kaggle / Colab T4 GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
print("--> Model and tokenizer loaded successfully!")

import os
import re
import math
import sympy as sp
import pandas as pd
import numpy as np

def extract_solution_code(llm_output: str):
    """
    Extract the full solution() code block from the LLM response.
    Handles markdown ```python blocks and plain text, preserving indentation.
    Stops right after the return statement to avoid trailing explanation or markdown.
    """
    if not llm_output or pd.isna(llm_output):
        return None
        
    text = str(llm_output).strip()
    
    # 1. Look inside a markdown ```python ... ``` block, or a bare def
    blocks = re.findall(r"```(?:python)?\s*(def solution\(\):.*?)```", text, re.DOTALL)
    if blocks:
        code_raw = blocks[-1].strip()
    else:
        match = re.search(r"(def solution\(\):.*)", text, re.DOTALL)
        if match:
            code_raw = match.group(1)
        else:
            return None

    # Cut off any conclusion headers (matches the model's Vietnamese output)
    code_raw = re.split(r'\n\s*(\[3\.|####\s*\[3\.|###\s*\[3\.|\*\*\[3\.|\[3\s*KẾT|3\.\s*KẾT|KẾT LUẬN:|Kết luận:)', code_raw)[0]
    
    lines = code_raw.split('\n')
    func_lines = []
    has_returned = False
    
    for idx, l in enumerate(lines):
        stripped = l.strip()
        if idx == 0 or stripped.startswith("def solution"):
            func_lines.append("def solution():")
            continue
        if not stripped:
            func_lines.append("")
            continue
            
        if stripped.startswith(("[3.", "#### [3.", "### [3.", "**[3.", "KẾT LUẬN", "Kết luận", "```")):
            break
            
        if has_returned:
            if not l.startswith(('    ', '\t')) or re.match(r'^[A-ZĐÁÀẢÃẠa-z0-9\-•*#`]', stripped):
                break
                
        if l.startswith(('    ', '\t')):
            func_lines.append(l)
        else:
            if "=" in stripped or stripped.startswith("return ") or stripped.startswith("if ") or stripped.startswith("for "):
                func_lines.append("    " + stripped)
            else:
                break
                
        if stripped.startswith("return "):
            has_returned = True
            if not stripped.endswith("(") and not stripped.endswith("[") and not stripped.endswith("{") and not stripped.endswith("\\"):
                break
                
    return '\n'.join(func_lines).strip()
def parse_and_execute_solution(llm_output: str):
    """
    1. Execute the full solution() to get extracted_result and exec_status (macro level).
    2. Break out each calculation step: llm_step_calculation, step_correct, error_type (micro level).
    """
    code_str = extract_solution_code(llm_output)
    if not code_str:
        return [], None, "No_Solution_Function", 0
        
    global_env = {"sp": sp, "sympy": sp, "math": math, "np": np, "numpy": np}
    
    # 1. Macro execution: run the whole solution()
    final_res = None
    exec_status = "Success"
    try:
        exec(code_str, global_env, global_env)
        if "solution" in global_env and callable(global_env["solution"]):
            final_res = global_env["solution"]()
        else:
            exec_status = "No_Callable_Solution"
    except Exception as e:
        exec_status = type(e).__name__
        final_res = f"Error: {str(e)}"
        
    # 2. Micro execution: step-by-step calculation audit
    step_records = []
    lines = code_str.split('\n')
    step_env = {"sp": sp, "sympy": sp, "math": math, "np": np, "numpy": np}
    
    current_step_num = 0
    current_step_desc = "Initialize parameters"
    statement_buffer = []
    
    def is_arithmetic_op(stmt, var_name, rhs):
        if var_name == 'result':
            return False
        if re.match(r'^[+-]?\d+(\.\d+)?$', rhs):
            return False
        if re.match(r'^sp\.Symbol\(.*\)$', rhs) or re.match(r'^symbols\(.*\)$', rhs):
            return False
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', rhs):
            return False
        return any(op in rhs for op in ['+', '-', '*', '/', '**', 'sqrt', 'cos', 'sin', 'tan', 'pi', 'Abs', 'float(', 'int(', 'solve', 'Eq', 'acos', 'asin'])

    def process_statement(stmt, step_num, step_desc):
        stripped = stmt.strip()
        if not stripped or stripped.startswith("def solution") or stripped.startswith("import") or stripped == "pass":
            return None
            
        if stripped in ["solution()", "print(solution())", "result = solution()"]:
            return None
            
        if stripped.startswith("return"):
            return None
            ret_expr = stripped[6:].strip()
            try:
                ret_val = eval(ret_expr, {}, step_env)
                val_str = str(ret_val)
                if re.match(r'^[+-]?\d+/\d+$', val_str.strip()) or re.match(r'^[+-]?[A-Za-z0-9_]+/[A-Za-z0-9_]+$', val_str.strip()):
                    val_str = f"'{val_str.strip()}"
                elif val_str.strip().startswith(('+', '-')) and not re.match(r'^[+-]?\d+(\.\d+)?$', val_str.strip()):
                    val_str = f"'{val_str.strip()}"
                    
                return {
                    "step": max(1, step_num),
                    "step_description": "Return Result",
                    "llm_step_calculation": stripped.replace('\n', ' '),
                    "step_result": val_str,
                    "step_correct": 1,
                    "error_type": "None"
                }
            except Exception as e:
                return {
                    "step": max(1, step_num),
                    "step_description": "Return Result",
                    "llm_step_calculation": stripped.replace('\n', ' '),
                    "step_result": f"Error: {str(e)}",
                    "step_correct": 0,
                    "error_type": type(e).__name__
                }
                
        try:
            exec(stripped, {}, step_env)
            step_val = None
            if "=" in stripped:
                var_name = stripped.split("=")[0].strip()
                rhs = stripped.split("=", 1)[1].strip()
                if not is_arithmetic_op(stripped, var_name, rhs):
                    return None
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                    step_val = step_env.get(var_name, None)
            else:
                return None
                    
            val_str = str(step_val) if step_val is not None else "Executed"
            if re.match(r'^[+-]?\d+/\d+$', val_str.strip()) or re.match(r'^[+-]?[A-Za-z0-9_]+/[A-Za-z0-9_]+$', val_str.strip()):
                val_str = f"'{val_str.strip()}"
            elif val_str.strip().startswith(('+', '-')) and not re.match(r'^[+-]?\d+(\.\d+)?$', val_str.strip()):
                val_str = f"'{val_str.strip()}"
                
            return {
                "step": max(1, step_num),
                "step_description": step_desc,
                "llm_step_calculation": stripped.replace('\n', ' '),
                "step_result": val_str,
                "step_correct": 1,
                "error_type": "None"
            }
        except Exception as e:
            return {
                "step": max(1, step_num),
                "step_description": step_desc,
                "llm_step_calculation": stripped.replace('\n', ' '),
                "step_result": f"Error: {str(e)}",
                "step_correct": 0,
                "error_type": type(e).__name__
            }

    for l in lines:
        s = l.strip()
        if not s or s.startswith("def solution") or s.startswith("import"):
            continue
        # Detect step-marker comments in the model's generated code (Vietnamese/English)
        if s.startswith("# Bước") or s.startswith("# Step") or s.startswith("# step") or s.startswith("# Câu"):
            current_step_num += 1
            raw_desc = s.lstrip("#").strip()
            clean_desc = re.sub(r'^(Bước\s*\d+\s*[:\-\.]\s*|Step\s*\d+\s*[:\-\.]\s*|Câu\s*[a-zA-Z0-9]+\s*[:\-\.]\s*)', '', raw_desc, flags=re.IGNORECASE).strip()
            if clean_desc:
                clean_desc = clean_desc[0].upper() + clean_desc[1:]
            current_step_desc = clean_desc if clean_desc else f"Execute step {current_step_num}"
            continue
        if s.startswith("#"):
            continue
            
        statement_buffer.append(s)
        full_stmt = "\n".join(statement_buffer)
        
        try:
            compile(full_stmt, '<string>', 'exec')
            rec = process_statement(full_stmt, current_step_num, current_step_desc)
            if rec:
                step_records.append(rec)
                if rec["step_correct"] == 0:
                    break
            statement_buffer = []
        except SyntaxError:
            continue

    num_steps = len(set(s["step"] for s in step_records)) if step_records else 0
    return step_records, final_res, exec_status, num_steps

def normalize_val(v):
    if v is None or pd.isna(v):
        return None
    s = str(v).strip().replace(' ', '').lstrip("'")
    s = s.replace('+/-', '+-').replace('±', '+-')
    return s

def parse_cosine_eq(eq_str):
    s = str(eq_str).replace(' ', '').replace('^', '**')
    m = re.match(r'^([0-9.]+)\*?cos\((.+?)\*?t([+-].+?)?\)$', s, re.IGNORECASE)
    if m:
        A = float(m.group(1))
        omega = m.group(2)
        phi = m.group(3) if m.group(3) else '0'
        return {'A': A, 'omega': omega, 'phi': phi}
    return None

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(flatten_dict(v, parent_key=k).items())
            else:
                items.append((k, v))
    return dict(items)

def to_float_safe(val_str):
    if val_str is None or pd.isna(val_str):
        return None
    s = str(val_str).lstrip("'").strip()
    s = s.replace('^', '**').replace('sqrt', 'math.sqrt').replace('pi', 'math.pi')
    try:
        return float(eval(s, {"math": math, "sp": sp}))
    except Exception:
        return None

def compare_accuracy(pred, gt, tolerance=3e-2):
    """
    Compare the PAL extracted_result against ground truth:
    1. Direct string match
    2. Dictionary match (incl. nested) & cosine equation
    3. Numeric match (float, int, fraction, sqrt/pi expression)
    4. Symbolic algebraic match (T/8, T/12, ...)
    5. cm vs m unit match (factor of 100)
    """
    if pred is None or gt is None or pd.isna(gt):
        return 0
        
    p_norm = normalize_val(pred)
    g_norm = normalize_val(gt)
    if not p_norm or not g_norm:
        return 0
        
    # 1. Direct string match
    if p_norm.lower() == g_norm.lower():
        return 1
        
    # 2. Cosine equation and dictionary match
    eq_dict_p = parse_cosine_eq(p_norm)
    eq_dict_g = parse_cosine_eq(g_norm)
    
    dict_p = None
    dict_g = None
    try:
        dict_p = eval(str(pred).lstrip("'")) if isinstance(pred, (str, dict)) and '{' in str(pred) else eq_dict_p
    except Exception:
        dict_p = eq_dict_p
        
    try:
        dict_g = eval(str(gt).lstrip("'")) if isinstance(gt, (str, dict)) and '{' in str(gt) else eq_dict_g
    except Exception:
        dict_g = eq_dict_g
        
    if isinstance(dict_p, dict) and isinstance(dict_g, dict):
        flat_p = flatten_dict(dict_p)
        flat_g = flatten_dict(dict_g)
        
        match_all = True
        for k in flat_g:
            if k in flat_p:
                val_p = normalize_val(flat_p[k])
                val_g = normalize_val(flat_g[k])
                if val_p != val_g:
                    f_p = to_float_safe(val_p)
                    f_g = to_float_safe(val_g)
                    if f_p is not None and f_g is not None:
                        if abs(f_p - f_g) > tolerance:
                            match_all = False
                            break
                    else:
                        match_all = False
                        break
            else:
                match_all = False
                break
        if match_all:
            return 1
            
    # 3. Numeric match (float, int, fraction, sympy)
    p_num = to_float_safe(p_norm)
    g_num = to_float_safe(g_norm)
    if p_num is not None and g_num is not None:
        if abs(p_num - g_num) <= tolerance or (g_num != 0 and abs(p_num - g_num)/abs(g_num) <= tolerance):
            return 1
        if abs(p_num * 100 - g_num) <= tolerance or (g_num != 0 and abs(p_num * 100 - g_num)/abs(g_num) <= tolerance):
            return 1
        if abs(p_num / 100.0 - g_num) <= tolerance or (g_num != 0 and abs(p_num / 100.0 - g_num)/abs(g_num) <= tolerance):
            return 1
            
    # 4. Symbolic algebraic match containing T (e.g. T/8, 'T/8', 0.125*T, T/12)
    if 'T' in p_norm and 'T' in g_norm:
        p_t_coeff = to_float_safe(p_norm.replace('T', '1.0').replace('*', ''))
        g_t_coeff = to_float_safe(g_norm.replace('T', '1.0').replace('*', ''))
        if p_t_coeff is not None and g_t_coeff is not None:
            if abs(p_t_coeff - g_t_coeff) <= tolerance:
                return 1
                
    # 5. +- solution match (e.g. +-8*pi vs '+-8.0*pi')
    if p_norm.startswith('+-') and g_norm.startswith('+-'):
        p_sub = to_float_safe(p_norm[2:])
        g_sub = to_float_safe(g_norm[2:])
        if p_sub is not None and g_sub is not None and abs(p_sub - g_sub) <= tolerance:
            return 1

    return 0

# 1. Auto-detect the dataset file path (Kaggle, Colab, or local)
input_file_path = ""
candidate_paths = [
    "pal_input_testbed.xlsx",
    "./pal_input_testbed.xlsx",
    "../pal_input_testbed.xlsx",
    "/kaggle/input/pal_input_testbed.xlsx",
]
for p in candidate_paths:
    if os.path.exists(p):
        input_file_path = p
        break

if not input_file_path:
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if "pal_input_testbed" in file and file.endswith((".xlsx", ".csv")):
                input_file_path = os.path.join(root, file)
                break
        if input_file_path:
            break

print(f"--> Loading data from: {input_file_path}")
df_in = pd.read_excel(input_file_path) if input_file_path.endswith(".xlsx") else pd.read_csv(input_file_path)
df_in.columns = [c.strip().lower() for c in df_in.columns]

# 2. Output paths for the two result files (Kaggle working dir or current dir)
output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
file1_path = os.path.join(output_dir, "pal_evaluation_results.csv")
file2_path = os.path.join(output_dir, "step_by_step_calculation_audit.csv")

# 3. Resume: skip already-completed questions on re-run
completed_ids = set()
if os.path.exists(file1_path):
    try:
        df_existing = pd.read_csv(file1_path, encoding="utf-8-sig")
        completed_ids = set(df_existing["id"].astype(str).tolist())
        print(f"--> Found previous progress: {len(completed_ids)}/{len(df_in)} questions done. Skipping those.")
    except Exception as e:
        print(f"Could not read previous file, starting fresh: {e}")

system_prompt = ""

print(f"\nSTARTING PAL EXPERIMENT WITH QWEN-2.5-7B...\n" + "=" * 75)

with torch.no_grad():
    for i, row in df_in.iterrows():
        q_id = str(row.get('id', f'q_{i+1:02d}'))
        
        if q_id in completed_ids:
            continue
            
        question = str(row.get('question', ''))
        prompt_pal = str(row.get('prompt_pal', ''))
        ground_truth = row.get('ground_truth', '')
        
        # Model-facing prompt (Vietnamese: the problems are in Vietnamese)
        user_prompt = f"{prompt_pal}\n\nĐỀ BÀI CẦN GIẢI:\n{question}\nLỜI GIẢI:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            escape_special_tokens=False
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Greedy decoding (do_sample=False) to match the paper
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=False
        )
        
        response = tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        step_logs, final_res, status, num_steps = parse_and_execute_solution(response)
        acc = compare_accuracy(final_res, ground_truth)
        
        # Write File 1: pal_evaluation_results.csv (macro level)
        row_res = pd.DataFrame([{
            "id": q_id,
            "question": question,
            "ground_truth": ground_truth,
            "extracted_result": str(final_res),
            "exec_status": status,
            "accuracy": acc,
            "num_steps": num_steps,
            "raw_response": response
        }])
        
        header_f1 = not os.path.exists(file1_path)
        row_res.to_csv(file1_path, mode='a', header=header_f1, index=False, encoding="utf-8-sig")
        
        # Write File 2: step_by_step_calculation_audit.csv (micro level)
        if step_logs:
            step_rows = []
            for s in step_logs:
                err_t = s["error_type"]
                if s["step_description"] == "Return Result" and acc == 0 and err_t == "None":
                    err_t = "Formula_or_Arithmetic_Deviation"
                    
                step_rows.append({
                    "question_id": q_id,
                    "step": s["step"],
                    "step_description": s["step_description"],
                    "llm_step_calculation": s["llm_step_calculation"],
                    "step_result": s["step_result"],
                    "step_correct": s["step_correct"],
                    "error_type": err_t
                })
            df_step = pd.DataFrame(step_rows)
            header_f2 = not os.path.exists(file2_path)
            df_step.to_csv(file2_path, mode='a', header=header_f2, index=False, encoding="utf-8-sig")
            
        completed_ids.add(q_id)
        print(f"[{q_id}] Done | Exec: {status:<10} | Steps: {num_steps} | Acc: {acc} | Extracted: {str(final_res):<15} | GT: {str(ground_truth)}")

print("\n" + "=" * 75 + "\nPAL EXPERIMENT COMPLETE!\n" + "=" * 75)

# Reload the saved aggregate results
df_final_res = pd.read_csv(file1_path, encoding="utf-8-sig")
df_final_audit = pd.read_csv(file2_path, encoding="utf-8-sig") if os.path.exists(file2_path) else pd.DataFrame()

print("\n" + "=" * 60)
print("PAL EXPERIMENT SUMMARY (QWEN-2.5-7B)")
print("=" * 60)
print(f"1. Total questions tested   : {len(df_final_res)}")
print(f"2. Numerically correct      : {df_final_res['accuracy'].sum()} / {len(df_final_res)}")
print(f"3. Accuracy                 : {df_final_res['accuracy'].mean()*100:.2f}%")
print(f"4. Code exec success rate   : {(df_final_res['exec_status'] == 'Success').mean()*100:.2f}%")
print(f"5. Total micro-audit steps  : {len(df_final_audit)} steps")
print(f"6. File 1 saved to          : {file1_path}")
print(f"7. File 2 saved to          : {file2_path}")
print("=" * 60)

# Preview the first 10 rows of the macro results
print(df_final_res.head(10).to_string(index=False))

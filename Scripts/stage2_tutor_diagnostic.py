#!/usr/bin/env python3

# STAGE 2 (FINAL): Virtual tutor on the balanced 62-submission set (31 faulty + 31 correct)
# Measures both sensitivity and specificity of the detection step -- i.e. does the model flag
# every submission as faulty?
# 31 faulty submissions: genuinely erroneous student work (Qwen role-playing a student).
# 31 correct submissions: s_idx=3 solutions (100% correct, human-graded) on the same questions.
# Machine-readable output @@VERDICT@@/@@STEP@@/@@FIX@@; the last block prints a confusion
# matrix comparing vote_verdict against ground_truth.
# Input: Dataset/stage2_input_62.xlsx (has a ground_truth column). Output: stage2_student_diagnostics_62.xlsx.

# Environment setup & loading Qwen-2.5-7B-Instruct
import os
import time
import random
import numpy as np
import pandas as pd
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

# Optimization: float16 + SDPA to maximize hardware throughput on Kaggle / Colab T4 GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
print("--> Model and tokenizer loaded successfully!")

# Virtual tutor V5 -- greedy T=0, machine-readable @@tag@@ output
import os, time, re, unicodedata
import pandas as pd, torch
def _N(s): return unicodedata.normalize("NFC", str(s if s is not None else ""))

input_file = "/kaggle/input/ágawbab/stage2_input_62.xlsx"
if not os.path.exists(input_file):
    for fb in ['stage2_input_62.xlsx','Dataset/stage2_input_62.xlsx']:
        if os.path.exists(fb): input_file=fb; break
print('--> input:', input_file)
df_input = pd.read_excel(input_file); print('--> submissions:', len(df_input))

output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
output_xlsx = os.path.join(output_dir, 'stage2_student_diagnostics_62.xlsx')
output_csv  = os.path.join(output_dir, 'stage2_student_diagnostics_62.csv')

# Model-facing tutor prompt (Vietnamese: the physics problems are in Vietnamese)
system_prompt = ("Bạn là Gia sư Vật lý lớp 11 (dao động điều hòa). Đối chiếu bài làm học sinh với Lời giải "
    "mẫu chuẩn 100%. So sánh theo GIÁ TRỊ toán học, không theo cách trình bày; đi tuần tự và chỉ dừng "
    "ở bước SAI THẬT SỰ đầu tiên (giá trị khác Lời giải mẫu). Không tự tính toán mới; mọi con số lấy "
    "từ Lời giải mẫu. In đúng 3 dòng: @@VERDICT@@, @@STEP@@, @@FIX@@.")

def parse_tags(fb):
    t=_N(fb)
    def grab(tag, nxt):
        m=re.search(r'@@'+tag+r'@@\s*:?\s*(.*?)(?=@@(?:'+nxt+r')@@|$)', t, re.I|re.S)
        return m.group(1).strip(" :\n\t*#>-") if m else ""
    vraw=grab('VERDICT','STEP|FIX')
    verdict='INCORRECT' if re.search(r'\bSAI\b',vraw,re.I) else ('CORRECT' if re.search(r'\u0110\u00daNG',vraw) else '?')
    return verdict, grab('STEP','FIX'), grab('FIX','')

def _gold_ans(std):
    nums=re.findall(r'-?\d+[.,]?\d*', _N(std).replace('\\',''))
    return nums[-1] if nums else ''

results=[]; start=time.time()
with torch.no_grad():
    for _, row in df_input.iterrows():
        msgs=[{"role":"system","content":system_prompt},
              {"role":"user","content":str(row.get("prompt",""))}]
        text=tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp=tokenizer([text], return_tensors="pt").to(model.device)
        gen=model.generate(**inp, max_new_tokens=1024, do_sample=False)
        feedback=tokenizer.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        v,step,fix=parse_tags(feedback)
        ga=_gold_ans(row.get("standard_solution",""))
        faithful_proxy=bool(ga) and (ga.replace(' ','') in _N(feedback).replace('\\','').replace(' ',''))
        results.append({
            "submission_id":row["submission_id"],"question_id":row["question_id"],"model":row.get("model",""),
            "question":row["question"],"standard_solution":row["standard_solution"],
            "student_solution":row.get("student_solution",""),"prompt":row.get("prompt",""),
            "tutor_feedback":feedback,"vote_verdict":v,"localization_text":step,"remedial_guidance":fix,
            "gold_final_answer":ga,"faithful_proxy":int(faithful_proxy),
            "ground_truth":row.get("ground_truth",""),
        })
        print(f'[{row["submission_id"]}|{row["question_id"]}] {v} | proxy={int(faithful_proxy)} | step={step[:45]}')
df_out=pd.DataFrame(results)
df_out.to_excel(output_xlsx, index=False)
df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
# parse health check
bad=(df_out['vote_verdict']=='?').sum(); empt=(df_out['localization_text'].str.len()<=3).sum()
print(f'SAVED: {output_xlsx} | time {time.time()-start:.1f}s | unparsed verdict={bad}/31 | empty step={empt}/31')

# CONFUSION MATRIX -- sensitivity / specificity / precision / accuracy
import pandas as pd
df = pd.read_excel(output_xlsx if 'output_xlsx' in locals() else 'stage2_student_diagnostics_62.xlsx')
n=len(df)
bad=(df['vote_verdict']=='?').sum()
print(f'Parsed OK: {n-bad}/{n} (parse fail={bad})')

# prediction: INCORRECT => model flags faulty; CORRECT => model passes as correct
pred_sai = df['vote_verdict']=='INCORRECT'
gt_sai   = df['ground_truth'].astype(str).str.strip().str.upper().eq('SAI')

TP=int(( gt_sai &  pred_sai).sum())   # faulty correctly caught
FN=int(( gt_sai & ~pred_sai).sum())   # faulty missed (passed as correct)
TN=int((~gt_sai & ~pred_sai).sum())   # correct correctly passed
FP=int((~gt_sai &  pred_sai).sum())   # correct wrongly flagged
def pct(a,b): return f'{100*a/b:.2f}%' if b else 'n/a'

print('\n================ CONFUSION MATRIX (verdict) ================')
print(f'                        pred FAULTY    pred CORRECT')
print(f'  actual FAULTY  (31):  TP={TP:<3}        FN={FN}')
print(f'  actual CORRECT (31):  FP={FP:<3}        TN={TN}')
print('-----------------------------------------------------------')
print(f'  Sensitivity/Recall (catch faulty)     : {pct(TP,TP+FN)}  ({TP}/{TP+FN})')
print(f'  Specificity        (pass correct)     : {pct(TN,TN+FP)}  ({TN}/{TN+FP})')
print(f'  Precision          (faulty predicted) : {pct(TP,TP+FP)}  ({TP}/{TP+FP})')
print(f'  Accuracy           (overall)          : {pct(TP+TN,n)}  ({TP+TN}/{n})')
prec=TP/(TP+FP) if TP+FP else 0; rec=TP/(TP+FN) if TP+FN else 0
print(f'  F1                                    : {2*prec*rec/(prec+rec):.3f}' if prec+rec else '  F1: n/a')

# list the mistakes for inspection
print('\nFN (missed faulty)       :', list(df[gt_sai & ~pred_sai]['submission_id']))
print('FP (correct wrongly flagged):', list(df[~gt_sai & pred_sai]['submission_id']))
print(df[['submission_id','question_id','ground_truth','vote_verdict']].to_string(index=False))

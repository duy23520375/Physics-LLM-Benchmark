#!/usr/bin/env python3

# STAGE 1: Generate canonical pedagogical solutions (22 questions)
# Project: Virtual Physics Tutor (for the SoICT paper)
# Model: Qwen/Qwen2.5-7B-Instruct (FP16 / SDPA, seed=42, do_sample=False).
# Task: turn the Step-2 def solution() code into a realistic, pedagogically standard worked solution.
# Output: Dataset/stage1_gold_solutions_22.xlsx & .csv (22 canonical solutions).

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

# Optimization: float16 to speed up inference on Kaggle / Colab T4 GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
print("--> Model and tokenizer loaded successfully!")

# Generate the 22 canonical pedagogical solutions (standard solutions)

possible_paths = [
    "stage1_input.xlsx",
    "stage1_input_22.xlsx",
    "Dataset/stage1_input.xlsx",
    "./stage1_input.xlsx",
    "./stage1_input_22.xlsx",
    "../stage1_input.xlsx",
    "../stage1_input_22.xlsx",
    "/kaggle/input/physics-pal-dataset/stage1_input_22.xlsx",
    "/kaggle/input/stage1_input.xlsx",
    "/kaggle/input/stage1_input_22.xlsx",
    "/content/stage1_input_22.xlsx",
]

input_file = None
for p in possible_paths:
    if os.path.exists(p):
        input_file = p
        break

if input_file is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        for file in files:
            if "stage1_input" in file and file.endswith(".xlsx"):
                input_file = os.path.join(root, file)
                break
        if input_file:
            break

if input_file is None:
    raise FileNotFoundError("Input file stage1_input_22.xlsx not found!")

print(f"--> Reading data from: {input_file}")
df_input = pd.read_excel(input_file)

output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
output_xlsx = os.path.join(output_dir, "stage1_gold_solutions_22.xlsx")

results = []
print(f"\nGENERATING CANONICAL PEDAGOGICAL SOLUTIONS FOR {len(df_input)} QUESTIONS...\n" + "=" * 75)
start_total = time.time()

with torch.no_grad():
    for idx, row in df_input.iterrows():
        q_id = row["question_id"]
        prompt_col = "convert_prompt" if "convert_prompt" in row else "verbalizer_prompt"
        prompt_text = str(row[prompt_col])

        # Model-facing system prompt (Vietnamese: solutions are written in Vietnamese)
        messages = [
            {"role": "system", "content": "Bạn là Giáo viên Vật lý chuyên nghiệp chuyên trình bày lời giải chi tiết, rõ ràng, chuẩn sư phạm cho học sinh lớp 11."},
            {"role": "user", "content": prompt_text}
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

        standard_solution = tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        results.append({
            "question_id": q_id,
            "question": row["question"],
            "script_result": row["script_result"],
            "convert_prompt": prompt_text,
            "standard_solution": standard_solution
        })

        print(f"[{q_id}] Done | generated solution ({len(standard_solution)} chars):")
        print(f"{standard_solution[:120]}...\n")

df_final = pd.DataFrame(results)
df_final.to_excel(output_xlsx, index=False)
output_csv = output_xlsx.replace('.xlsx', '.csv')
df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')
print("=" * 75)
print(f"STAGE 1 COMPLETE! SAVED 22 CANONICAL SOLUTIONS TO: {output_xlsx}")
print(f"Total run time: {time.time() - start_total:.2f}s")

# Preview the generated canonical solutions
import pandas as pd

df_res = pd.read_excel(output_xlsx if 'output_xlsx' in locals() else 'stage1_gold_solutions_22.xlsx')
print(f"Total canonical solutions: {len(df_res)} questions\n")

for idx, r in df_res.head(5).iterrows():
    print("="*80)
    print(f"[{r['question_id']}] QUESTION: {r['question']}")
    print(f"CANONICAL PYTHON ANSWER: {r['script_result']}")
    print(f"---\nCANONICAL PEDAGOGICAL SOLUTION:\n{r['standard_solution']}")

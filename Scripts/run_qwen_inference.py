import csv
import glob
import os
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. SYSTEM CONFIGURATION & SEED SETUP (REPRODUCIBILITY)
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# 2. MODEL VERSION & DATA TYPE CONFIGURATION
# ==========================================
# Model selection: Uncomment the model you want to run:
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Small model 1.5B
model_name = "Qwen/Qwen2.5-3B-Instruct"    # Medium model 3B (Default)
# model_name = "Qwen/Qwen2.5-7B-Instruct"    # Large model 7B (Best for physical reasoning)

print(f"Loading tokenizer and model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Data type configuration (torch_dtype) depending on your GPU:
# - torch.float16: For older GPUs not supporting bfloat16 (e.g. Tesla T4, P100 on Kaggle). Runs extremely fast on T4.
# - torch.bfloat16: For newer GPUs supporting bfloat16 (e.g. A100, RTX 30xx, RTX 40xx).
dtype = torch.float16  # Default is float16 to optimize run speed on Kaggle GPU T4

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",
    attn_implementation="sdpa"  # Enable SDPA for faster token generation
)

# ==========================================
# 3. PROJECT DIRECTORY CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "Results", "foundational_testbed_40_results.csv")
PROMPTS_DIR = os.path.join(BASE_DIR, "Prompts")

def get_prompt_template(question_num):
    """Locate and read the full prompt file from Prompts/ directory based on question index"""
    prompt_pattern = os.path.join(PROMPTS_DIR, f"{question_num}_*.txt")
    prompt_files = glob.glob(prompt_pattern)
    if not prompt_files:
        return None
    with open(prompt_files[0], 'r', encoding='utf-8') as pf:
        template = pf.read().strip()
    return template.strip('"""').strip("'''").strip()

# ==========================================
# 4. RUN INFERENCE ON THE DATASET
# ==========================================
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset not found at {DATASET_PATH}")
    exit(1)

# Read all questions from Results/foundational_testbed_40_results.csv
questions = []
with open(DATASET_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        questions.append(row["question"])

print(f"Loaded {len(questions)} questions from dataset. Starting evaluation...")
print("=" * 60)

with torch.no_grad():
    for idx, question in enumerate(questions):
        q_num = idx + 1
        prompt_template = get_prompt_template(q_num)
        
        if not prompt_template:
            print(f"Skipping question {q_num}: Prompt file not found.")
            continue
            
        messages = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": question}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            escape_special_tokens=False
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Use Greedy decoding to ensure reproducibility
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048, 
            do_sample=False
        )

        response = tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        print(f"\nQUESTION {q_num}: {question}")
        print("-" * 30)
        print(response)
        print("\n" + "=" * 60)

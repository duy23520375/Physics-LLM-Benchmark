import csv
import os

csv_path = os.path.join(os.path.dirname(__file__), "..", "Results", "foundational_testbed_40_results.csv")

if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    exit(1)

with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Column indices for evaluation metrics in foundational_testbed_40_results.csv
models = {
    "GPT-4o (Zero-shot)": {"s_idx_col": 3, "comp_col": None},
    "Gemini-2.5-Flash (Zero-shot)": {"s_idx_col": 6, "comp_col": None},
    "Qwen-1.5B (Prompted)": {"s_idx_col": 9, "comp_col": 10},
    "Qwen-3B (Prompted)": {"s_idx_col": 13, "comp_col": 14},
    "Qwen-7B (Prompted)": {"s_idx_col": 17, "comp_col": 18}
}

print("=== REPRODUCING TABLE II: OVERALL PERFORMANCE ===")
print(f"{'Model':<30} | {'Acc (%)':<8} | {'S_idx (%)':<10} | {'C_rate (%)':<10}")
print("-" * 68)

for name, cols in models.items():
    s_col = cols["s_idx_col"]
    c_col = cols["comp_col"]
    
    total = len(rows)
    correct = 0
    total_score = 0
    compliant = 0
    
    for row in rows:
        score = int(row[s_col].strip())
        total_score += score
        if score == 3:
            correct += 1
            
        if c_col is not None:
            if row[c_col].strip() == "1":
                compliant += 1
                
    acc = (correct / total) * 100
    s_idx = (total_score / (total * 3)) * 100
    c_rate = (compliant / total) * 100 if c_col is not None else "--"
    
    c_rate_str = f"{c_rate:.1f}" if isinstance(c_rate, float) else c_rate
    print(f"{name:<30} | {acc:<8.1f} | {s_idx:<10.1f} | {c_rate_str:<10}")

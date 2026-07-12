import csv
import os

csv_path = os.path.join(os.path.dirname(__file__), "..", "Results", "arithmetic_stress_test_450_results.csv")

if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
    exit(1)

with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

levels = ["Easy", "Medium", "Hard"]
counts = {lv: 0 for lv in levels}
corrects = {lv: 0 for lv in levels}

for row in rows:
    level = row[0].strip()
    acc_val = float(row[4].strip())
    counts[level] += 1
    if acc_val == 1.0:
        corrects[level] += 1

print("=== REPRODUCING TABLE IV: STRESS TEST RESULTS ===")
print(f"{'Level':<10} | {'Correct/Total':<15} | {'Accuracy (%)':<15}")
print("-" * 46)

for lv in levels:
    cnt = counts[lv]
    corr = corrects[lv]
    pct = (corr / cnt) * 100 if cnt > 0 else 0
    print(f"{lv:<10} | {f'{corr}/{cnt}':<15} | {pct:<15.1f}")

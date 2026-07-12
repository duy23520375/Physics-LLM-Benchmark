# A Preliminary Study on Building an LLM-Powered Physics AI Tutor for High School Students

This repository contains the official dataset, prompt templates, response logs, and evaluation scripts for the paper: **"A Preliminary Study on Building an LLM-Powered Physics AI Tutor for High School Students"** (MAPR 2026).

---

## Abstract

Applying Large Language Models (LLMs) to Physics remains challenging due to the complex integration of domain theory, real-world modeling, and precise computation. This paper presents a pre-feasibility study evaluating LLMs as physics AI tutors, focusing on prompting strategies and computational robustness. We curated an expert-validated, anti-leakage testbed of 442 open-ended Mechanical Oscillation problems and applied a *Constrained Prompt Engineering* technique (Theory - Calculation - Conclusion) to systematically investigate model behavior. 

Empirical results, evaluated via a 4-level rubric isolating reasoning from calculation errors, reveal a critical dichotomy: while constrained models under a structured paradigm (e.g., Qwen2.5-7B-Instruct) exhibit excellent logical reasoning (96.7% reasoning capability index), they face severe arithmetic barriers. Specifically, a diagnostic stress-test on 450 mathematical variants shows accuracy plummeting from 75.3% to 33.3% when processing complex irrationals. Furthermore, maintaining logical consistency requires excessively long prompts, posing context overhead challenges. These findings establish a preliminary baseline for LLM-based tutoring, suggesting Supervised Fine-Tuning (SFT) to internalize physical logic and Tool-use integration to resolve mathematical blind spots.

---

## Project Structure

```text
Physics-LLM-Benchmark/
│
├── Dataset/                                  # Pristine Benchmark Datasets
│   └── physics_testbed_curation.csv          # Core dataset of 442 curated physics questions & answers
│
├── Prompts/                                  # Constrained Few-Shot Prompt Templates
│   ├── 1_Kinematics_Synthesis.txt            # Few-shot prompt for Question 1
│   ├── 2_Kinematics_Synthesis.txt
│   └── ... (all 40 prompt template files)
│
├── Results/                                  # Comparative Logs & Human Scores
│   ├── foundational_testbed_40_results.csv   # Responses & s_idx scores of all models on 40 benchmark questions
│   ├── prompt_tuning_variants_80_results.csv # Model outputs on prompt tuning variants (80 questions)
│   └── arithmetic_stress_test_450_results.csv# Qwen-7B outputs on 450 stress-test mathematical variants
│
├── scripts/                                  # Python Inference & Reproducibility Scripts
│   ├── run_qwen_inference.py                 # Runs Qwen-Instruct inference (Interactive & Batch modes)
│   ├── calculate_metrics.py                  # Computes Accuracy, S_idx, and C_rate (Table II & III)
│   └── calculate_stress_test.py              # Computes Arithmetic Stress-Test Accuracies (Table IV)
│
├── LICENSE                                   # Repository license (MIT License)
└── README.md                                 # Project documentation
```

---

## Setup & Requirements

To run the inference and metric calculation scripts, you need Python 3.8+ with PyTorch and Hugging Face Transformers installed.

```bash
# Clone the repository
git clone https://github.com/duy23520375/Physics-LLM-Benchmark.git
cd Physics-LLM-Benchmark

# Install dependencies
pip install torch transformers numpy
```

---

## How to Reproduce Results

### 1. Reproducing Table II & Table III (Overall Performance & Metrics)
To calculate the model accuracies, reasoning capability indices ($S_{idx}$), and constraint compliance rates ($C_{rate}$) across GPT-4o, Gemini 2.5 Flash, Qwen-1.5B, Qwen-3B, and Qwen-7B:

```bash
python scripts/calculate_metrics.py
```

### 2. Reproducing Table IV (Arithmetic Stress-Test)
To calculate the performance of the Qwen-7B model across the Easy, Medium, and Hard tiers of the mathematical variants stress test:

```bash
python scripts/calculate_stress_test.py
```

### 3. Running Model Inference (Qwen)
You can run model inference locally or on Kaggle/Colab using the provided evaluation script. It supports two modes:
1.  **Interactive Mode:** Allows you to manually type and test any physics question.
2.  **Batch Mode:** Reads the questions and prompt templates dynamically from the dataset and executes them sequentially.

```bash
python scripts/run_qwen_inference.py
```
*Note: The script uses `torch.float16` by default to optimize inference speed on Kaggle's Tesla T4 GPUs.*

---

## Citation

If you find our work, dataset, or scripts helpful, please cite our paper:

```bibtex
@inproceedings{duy2026physicstutor,
  title={A Preliminary Study on Building an LLM-Powered Physics AI Tutor for High School Students},
  author={Duy, Nguyen N. M. and et al.},
  booktitle={Proceedings of the International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  year={2026}
}
```

# A Pre-feasibility Study of Applying Large Language Models to High School Physics: Evaluating Limitations and the Effectiveness of Constrained Prompting

This repository contains the dataset, prompt engineering framework, and evaluation results for our research on the performance of Large Language Models (LLMs) in solving Grade 11 Physics problems (Vietnamese Curriculum).

---

## Abstract
While Large Language Models (LLMs) have demonstrated impressive mathematical reasoning capabilities, applying them to solve Physics problems remains a significant challenge. Physics requires a complex integration of specialized theoretical knowledge, real-world modeling, and precise arithmetic calculations. This study serves as a pre-feasibility research to evaluate the current capabilities and limitations of LLMs in high school Physics education, laying the groundwork for future specialized fine-tuning processes. We introduce an open-ended dataset consisting of 442 problems on "Mechanical Oscillations." Based on this dataset, we propose the **Constrained Prompt Engineering** method, which enforces LLMs to adhere to a three-step reasoning structure (Theory - Calculation - Conclusion) along with strict physical constant constraints. Combined with a 4-level rubric to decouple reasoning errors from calculation errors, experimental results show that our technique enables a small-scale open-source model (Qwen-7B-Instruct) to achieve a reasoning score ($S_{idx}$) of up to 96.7%, surpassing the performance of massive commercial models such as GPT-5.3 (83.3%) and Gemini 3 Flash (88.3%).

---

## Methodology: Three-Step Constrained Prompting
Our framework enforces a strict logic flow by embedding physical and mathematical constraints directly into the following stages:

1. **[1. Theory & Setup]**: Requires a concise explanation (1-2 lines) of the core physics phenomenon or definition. For equation-based problems, the model is forced to define the standard form $x = A \cos(\omega t + \phi)$. If the input uses Sine or negative Cosine, the model must explicitly state the conversion process to the standard Cosine form before proceeding.
2. **[2. Calculation & Substitution]**: The mathematical execution stage. The model must first present the general formula using variables (e.g., $f = N/t$, $v_{max} = \omega A$), followed by detailed numerical substitution and unit conversion to the SI system (m, s, rad). For simpler tasks (e.g., calculating phase at time $t$), merging expression identification and substitution is permitted to optimize response length.
3. **[3. Conclusion]**: Presents only the final calculated value from Step 2, accompanied by the correct physical units.

---

## Project Structure
```text
├── Dataset/                # Full dataset of 442 physics problems
│   └── physics_442_dataset.csv
├── Prompts/                
│   └── final_benchmark_40/ # 40 validated prompts for final benchmarking
├── Results/                # Human-annotated results and comparative analysis
│   └── Detailed_Results.xlsx
└── README.md
```
---
## Citation
If you find this research or dataset helpful, please cite our work as follows:

```bibtex
@inproceedings{nguyen2026physics,
  title={A Pre-feasibility Study of Applying Large Language Models to High School Physics: Evaluating Limitations and the Effectiveness of Constrained Prompting},
  author={Nguyen, Huu Khanh Duy and Nguyen, Trong Chinh},
  booktitle={2026 IEEE International Conference (UIT-VNU)},
  year={2026},
  organization={IEEE}
}

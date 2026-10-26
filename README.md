# Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2608.26587-b31b1b.svg)](https://arxiv.org/abs/2608.26587)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue.svg)](https://arxiv.org/abs/2608.26587)

Code for the paper **"Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models"** (Saksham Khatwani, He Cheng, Majid Afshar, Dmitriy Dligach, Yanjun Gao), from the [LARK NLP Lab](https://www.larknlp.com/).

## Overview

How should biomedical knowledge graphs (KGs) be taught to an LLM that will be used for clinical diagnosis? This work compares **five KG task formulations**, **three training paradigms**, **two KGs** (UMLS and PrimeKG) and **three base LLMs** (Gemma-7B, Qwen2.5-7B-Instruct, Qwen3-8B), with patient progress notes as the clinical context.

Every paradigm improves on the non-finetuned baseline, but paradigms with similar in-domain accuracy transfer very differently. To explain why, the paper introduces two measures of *how* training changes a pretrained model:

- **Gradient Intervention Density (GID)**: the fraction of layer components whose change exceeds a small threshold δ.
- **Gradient Distortion (GD)**: 1 − cosine similarity between a layer's gradient and the pretrained anchor.

**Key finding.** KG-judgment training under KL regularization produces sparse, localized, direction-preserving updates, which we call **surgical alignment**. Task-specific SFT produces dense updates. A controlled ablation attributes the sparsity to both the training objective and the KL term, and the sparse-update paradigms tend to give better clinical reasoning quality (scored with the PDSQI-9 rubric) even when their in-domain accuracy is lower than task SFT's. See the paper for full results, including the cases where this pattern does not hold.

## Repository structure

```
.
├── sft_scripts/
│   ├── main.py                        # LoRA SFT with a KL penalty to the frozen base model
│   └── kldiv_hyperparam_tuning.py     # Optuna search over the KL coefficient λ
├── grpo_scripts/
│   ├── grpo_train_script.py           # GRPO with binary task rewards
│   └── grpo_train_script_soft_rewards.py  # GRPO with signed rewards + path-alignment reward
├── evaluation_scripts/
│   └── evaluate_models.py             # In-domain evaluation on the KG tasks (ROUGE, CUI-F1)
└── pdsqi-9/
    ├── extract_llm_output.py          # Generate diagnostic reasoning on ProbSum test notes
    └── pdsqi-9-eval.py                # LLM-as-judge scoring with the PDSQI-9 rubric
```

## KG task formulations

Each example pairs a patient progress note with KG paths written as `Node1->Relation|Node2->Relation|Node3...`.

| Task | Description | `task` field (GRPO) | `--eval_metric` |
|---|---|---|---|
| **P@10** | Pick the one valid path out of 10 candidates | `Path_Selection` | `d1` |
| **P@2** | Pick the valid path out of 2 candidates | `Path_Selection` | `d2` |
| **NHP** | Next-hop prediction: given a partial path, predict the next node or relation | `NHP` | `d3` |
| **PC** | Partial completion: complete the rest of a partial path | *(any other value)* | `d4` |
| **PN@10** | Select *all* valid paths out of 10 candidates | `Multi_Path_Selection` | `d5` |

`evaluate_models.py` also has `mcq` (4-way path multiple choice) and a fallback binary YES/NO path-validity mode.

## Setup

### Requirements

Python **3.12+** is required (the scripts use nested quotes inside f-strings). A CUDA GPU is required; models are loaded in 4-bit NF4 via bitsandbytes and trained with LoRA.

```bash
pip install torch transformers trl peft bitsandbytes accelerate datasets \
            evaluate rouge_score nltk scikit-learn spacy pandas pyyaml python-dotenv \
            optuna vllm quickumls simstring-fast openai azure-identity
```

### Environment variables

Create a `.env` file in the directory you run scripts from:

```bash
HUGGING_FACE_LOGIN_TOKEN=hf_...
# Only needed for PDSQI-9 judging
AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_TOKEN_ENDPOINT=https://cognitiveservices.azure.com/.default
```

### External resources

These are not distributed with the repository and must be obtained separately:

| Resource | Used by | Expected location |
|---|---|---|
| **ProbSum** (BioNLP 2023 Shared Task 1A, derived from MIMIC-III; requires PhysioNet credentialed access) | evaluation, PDSQI-9 | `../BioNLP2023-1A-Train.csv`, `../../BioNLP2023-1A-Test.csv` |
| **QuickUMLS index** built from a UMLS release (requires a UMLS license) | evaluation, GRPO rewards | `../quickumls/` |
| **PrimeKG** `kg.csv` | GRPO rewards (`--is_prime 1`) | `./primeKG/kg.csv` |
| CUI → entity name dictionary (`cui_to_entity.pkl`) | GRPO NHP reward fallback | set `cui_to_entity_path` in the GRPO scripts |

Paths are relative to the working directory and hard-coded at the top of each script; edit them to match your layout.

### Training data format

Training and evaluation data are pickled lists of dicts. The fields used by the scripts are:

| Field | Meaning |
|---|---|
| `prompt` | Full model input (instructions, progress note, candidate/partial paths) |
| `answer` | Gold completion (a path, a next hop, or a path suffix) |
| `progress_note` | The patient progress note (evaluation) |
| `label_path` / `label_paths` | Gold full path(s) |
| `wrong_options` | Distractor paths |
| `task` | `Path_Selection`, `Multi_Path_Selection`, `NHP`, or other (PC) — GRPO only |
| `source`, `id`, `prediction_type` | Metadata |

## Usage

### 1. KL-regularized SFT

LoRA (r=16, α=32) on the attention projections, with loss `L = L_SFT + λ · KL(p_base ‖ p_θ)`, where the reference distribution comes from the same model with the adapter disabled.

```bash
cd sft_scripts
python main.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --input_dataset ./data/train.pkl \
  --val_dataset ./data/val.pkl \
  --output_dir ./models/Qwen7B_SFT_d1 \
  --learning_rate 3e-5 \
  --kl_lambda 0.05 \
  --max_len 4200
```

Set `--kl_lambda 0` to train without the KL term. To tune λ (Optuna, 10 trials, log-uniform over [0.01, 1], minimizing eval loss):

```bash
python kldiv_hyperparam_tuning.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --input_dataset ../final_training_data/train_d1.pkl \
  --val_dataset ../final_training_data/train_val_d1.pkl \
  --output_dir ./hyperparam_search/Qwen7B_SFT_d1
```

> Both SFT scripts import `KLDivDataset` from a `KLDiv_dataset` module, which you need on your `PYTHONPATH`.

### 2. GRPO

Starts from a base model or from an SFT LoRA checkpoint (if `--model_name` is a local directory, its latest checkpoint is loaded and training continues on the adapter). Uses vLLM in colocate mode, 6 generations per prompt, β = 0.001, and early stopping on eval reward.

```bash
cd grpo_scripts
python grpo_train_script.py \
  --model_name ./models/Qwen7B_SFT_d1/ \
  --input_dataset ./grpo_data/train.pkl \
  --val_dataset ./grpo_data/val.pkl \
  --output_dir ./models/GRPO/Qwen7B_GRPO_d1 \
  --learning_rate 1e-6 \
  --is_prime 0        # 1 = PrimeKG (SimString matching), 0 = UMLS (QuickUMLS matching)
```

Rewards, computed on the text inside `<Answer>...</Answer>`:

- **Path selection (P@2, P@10, PN@10):** 1 if every gold path is included and no distractor is, else 0.
- **NHP:** relation hops must match exactly; node hops are compared by concept ID (UMLS CUI via QuickUMLS, or PrimeKG ID via SimString).
- **PC:** 1 if the gold suffix appears in the answer.
- Missing `<Answer>` tags give 0.

`grpo_train_script_soft_rewards.py` uses the same CLI but replaces these with signed rewards (+0.1 correct, −1 incorrect or malformed) and adds a second **path-alignment reward**: token coverage of the gold path in the completion, scaled by a repetition penalty and capped at 1.5.

### 3. In-domain evaluation

```bash
cd evaluation_scripts
python evaluate_models.py \
  --model_names Qwen/Qwen2.5-7B-Instruct ./models/Qwen7B_SFT_d1 ./models/GRPO/Qwen7B_GRPO_d1 \
  --data_file ./data/test_d1.pkl \
  --eval_metric d1 \
  --with_diagnosis 0
```

Pass several `--model_names` to evaluate them in sequence; local directories are loaded as LoRA adapters. Path tasks report ROUGE and **CUI-F1** (precision/recall/F1 between UMLS concepts extracted from the predicted and gold paths). `--with_diagnosis 1` adds the gold diagnoses to the prompt.

### 4. Reasoning quality (PDSQI-9)

Generate free-text diagnostic reasoning for ProbSum test notes, then score it with an LLM judge on the nine PDSQI-9 criteria (citations, accuracy, thoroughness, usefulness, organization, comprehensibility, succinctness, synthesis, stigmatizing language).

```bash
cd pdsqi-9
python extract_llm_output.py \
  --model_name ./models/Qwen7B_SFT_d1 \
  --output_dir ./outputs/qwen7b_sft_reasoning.pkl

python pdsqi-9-eval.py \
  --input_file ./outputs/qwen7b_sft_reasoning.pkl \
  --output_dir ./outputs/qwen7b_sft_pdsqi9.json \
  --error_path ./outputs/qwen7b_sft_errors.pkl
```

The judge is called through Azure OpenAI; set the deployment name in `pdsqi-9-eval.py` to match your Azure resource. Responses that fail to parse as JSON after one retry are written to `--error_path`.

## Citation

If you use this code, please cite:

```bibtex
@misc{khatwani2026surgical,
  title         = {Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models},
  author        = {Khatwani, Saksham and Cheng, He and Afshar, Majid and Dligach, Dmitriy and Gao, Yanjun},
  year          = {2026},
  eprint        = {2608.26587},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.26587}
}
```

## Data use

ProbSum is derived from MIMIC-III, and UMLS requires a license. No patient data or UMLS content is included in this repository. Follow the PhysioNet data use agreement and the UMLS license terms when reproducing these experiments.

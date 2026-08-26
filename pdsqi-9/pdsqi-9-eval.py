from openai import OpenAI
import os


api_key = "" #put int the azure key
# Initialize client with your API key
#client = OpenAI(api_key=key) 

import os
import json
from textwrap import dedent
from openai import OpenAI
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import pickle
import argparse
from huggingface_hub import login
import time
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/Qwen7B_SFT_d1', help='output directory')
    parser.add_argument('--error_path', type=str, default='default_error.pkl', help='error directory')
    args = parser.parse_args()
    return args


# =============== FULL PDSQI-9 RUBRIC (authoritative, JSON-ready) ===============
# This mirrors your nine criteria, definitions, and scale guidance.
PDSQI9_RUBRIC = {
    "metadata": {
        "name": "Provider Documentation Quality & Summarization Instrument (PDSQI-9)",
        "version": "2024-10-15",
        "note": "Evaluate AI-generated diagosis reasoning for each patient progress notes",
        "scale": "1–5 Likert (1=Not at all, 5=Extremely)"
    },
    "criteria": [
        {
            "id": "citations",
            "label": "Are citations present and appropriate?",
            "description": "Every assertion that needs support is correctly cited and prioritized by relevance.",
            "scale_notes": {
                "1": "Multiple incorrect citations OR no citations provided",
                "2": "One citation incorrect OR citations grouped together and not with individual assertions",
                "3": "Citations correct but some assertions missing a citation",
                "4": "Every assertion correctly cited with some relevance prioritization",
                "5": "Every assertion is correctly cited and prioritized by relevance"
            }
        },
        {
            "id": "accuracy_extractive",
            "label": "Is the reasoning accurate in extraction (extractive reasoning)?",
            "description": (
                "Reasoning facts must faithfully match source note text. Penalize incorrect information via "
                "fabrication (invented facts) or falsification (distorted facts)."
            ),
            "definitions": {
                "fabrication": "Invented information not in source notes.",
                "falsification": "Changing critical details so statements are no longer true per source.",
                "notes": (
                    "If the provider statement in the note is itself clinically wrong, but the summary faithfully repeats "
                    "it from the source, that is NOT fabrication or falsification by the summarizer."
                )
            },
            "scale_notes": {
                "1": "Multiple major errors with overt falsifications or fabrications",
                "2": "A major error in an assertion with overt falsification or fabrication",
                "3": "At least one assertion misaligned to context/timing/specificity though drawn from source",
                "4": "At least one assertion misaligned in source/timing but still factual (e.g., diagnosis/treatment)",
                "5": "All assertions can be traced back to the notes"
            }
        },
        {
            "id": "thoroughness",
            "label": "Is the reasoning thorough without any omissions?",
            "description": (
                "Covers all critical patient issues. Identify omissions relative to the use case and intended user."
            ),
            "omission_types": {
                "pertinent": "Essential for decisions/actions; missing it may directly impact care.",
                "potentially_pertinent": "Useful for understanding; may not directly alter current decisions."
            },
            "scale_notes": {
                "1": "More than one pertinent omission occurs",
                "2": "One pertinent and multiple potentially pertinent omissions occur",
                "3": "Only one pertinent omission occurs",
                "4": "Some potentially pertinent omissions occur",
                "5": "No pertinent or potentially pertinent omissions occur"
            }
        },
        {
            "id": "usefulness",
            "label": "Is the reasoning useful?",
            "description": "Information is pertinent and at an appropriate level of detail for the intended audience.",
            "scale_notes": {
                "1": "No assertions are pertinent to the target user",
                "2": "Some assertions are pertinent to the target user",
                "3": "Assertions pertinent but level of detail inappropriate (too detailed or not enough)",
                "4": "No non‑pertinent assertions but some are only potentially pertinent",
                "5": "No non‑pertinent assertions and level of detail appropriate to targeted user"
            }
        },
        {
            "id": "organization",
            "label": "Is the reasoning organized?",
            "description": "Logical order and grouping (temporal or systems/problem-based) supporting understanding.",
            "scale_notes": {
                "1": "Assertions out of order; groupings incoherent (completely disorganized)",
                "2": "Some assertions out of order OR grouping incoherent",
                "3": "No change in order/grouping from original input",
                "4": "Logical order OR grouping (not both) throughout",
                "5": "Logical order AND grouping throughout (completely organized)"
            }
        },
        {
            "id": "comprehensibility",
            "label": "Is the reasoning comprehensible with clarity of language?",
            "description": "Clear, plain language; avoid ambiguity and unfamiliar terminology for the target user.",
            "scale_notes": {
                "1": "Overly complex/inconsistent with unfamiliar terminology to target user",
                "2": "Any use of overly complex/inconsistent/unfamiliar terminology",
                "3": "Unchanged complex terms when simplification possible",
                "4": "Some improvement in structure/terminology toward clarity",
                "5": "Plain, familiar, well‑structured throughout"
            }
        },
        {
            "id": "succinctness",
            "label": "Is the reasoning succinct with economy of language?",
            "description": "Brief, to the point; avoid syntactic or semantic redundancy.",
            "scale_notes": {
                "1": "Too wordy across assertions with redundancy in syntax and semantics",
                "2": "More than one assertion with contextual semantic redundancy",
                "3": "At least one assertion with semantic redundancy OR multiple syntactic redundancies",
                "4": "No syntax redundancy; at least one could be shorter semantically",
                "5": "Fewest words possible; no redundancy"
            }
        },
        {
            "id": "synthesis_abstraction",
            "label": "Is abstraction needed, and if so, how well is it synthesized?",
            "description": (
                "If abstraction is needed, paraphrase/integrate facts into higher‑level clinically relevant reasoning. "
                "If no abstraction is needed, score as 'N/A'."
            ),
            "scale_notes": {
                "1": "Incorrect reasoning or grouping among assertions",
                "2": "Abstraction performed when not needed OR inappropriate grouping",
                "3": "Missed opportunity to abstract (facts stated independently)",
                "4": "Grouped into themes; limited but relevant clinical reasoning",
                "5": "Fully integrated clinical synopsis with prioritized information"
            },
            "applicability": "Use 'N/A' if the task does not require abstraction."
        },
        {
            "id": "stigmatizing_language",
            "label": "Is there presence of stigmatizing language?",
            "description": (
                "Avoid discrediting/judgmental terms (e.g., 'claims', 'insists', 'reportedly'); prefer person‑first "
                "language; minimize blame; avoid punitive connotations."
            ),
            "checks": [
                "present_in_source_note (bool)",
                "present_in_summary (bool)"
            ],
            "examples_note": (
                "Use 'patient with diabetes' rather than 'diabetic patient'; avoid 'addict', 'abuser'; avoid phrases that "
                "imply disbelief or blame."
            )
        }
    ]
}

# 2) Build prompt (concise instructions + optional full rubric injection)
CONDENSED_RUBRIC_TEXT = dedent("""
You are an expert clinical diagnosis evaluator using the PDSQI-9 rubric.
You are provided with PROGRESS_NOTES of the patient, the DIAGNOSIS of the patient and LLM_DIAGNOSIS_REASONING. 
DIAGNOSIS_REASONING represents the AI generated diagnosis prediction with reasoning. You need to judge it based on the PDSQI-9 rubric.
Score each criterion 1–5 (or "N/A" only for Synthesis/Abstraction when not applicable). Use short, clinical rationales.

CRITERIA (condensed):
- Citations: every needed assertion is correctly cited and prioritized by relevance.
- Accuracy (extractive): facts must faithfully match the source; penalize falsification (distorted facts) and fabrication (invented facts).
- Thoroughness: no pertinent or potentially pertinent omissions relative to the intended use.
- Usefulness: content is pertinent and at the right level of detail for the intended audience.
- Organization: logical temporal or problem-based order AND grouping.
- Comprehensibility: plain, unambiguous language for the target user.
- Succinctness: fewest words, no redundancy.
- Synthesis/Abstraction: integrate into higher-level reasoning if needed; otherwise mark "N/A".
- Stigmatizing language: flag discrediting/judgmental terms; prefer person-first language.

TASK:
1) Quote exact evidence spans from PROGRESS_NOTE that support/contradict LLM_DIAGNOSIS_REASONING assertions.
2) List omissions (pertinent vs potentially pertinent).
3) Provide a brief rationale for each score.
4) Produce an overall verdict: "safe_for_use" (yes/no) with one-sentence justification.

Rules: Return ONLY valid JSON (no prose, no backticks). Scores must be integers 1–5 except Synthesis/Abstraction may be "N/A".
""").strip()
# =====================================

# 0) Setup client
#api_key = os.getenv("OPENAI_API_KEY")
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), os.getenv("AZURE_TOKEN_ENDPOINT")
)

deployment_name = "gpt-5-mini"
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    azure_ad_token_provider=token_provider,
)

def run_eval(prompt_text: str, model) -> dict:
    for attempt in range(2):
        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            max_completion_tokens=40000,
            model=model
        )

        try:
            # print("output_text", resp.choices[0].message)
            return json.loads(resp.choices[0].message.content), 0
        except Exception as e:
            if attempt == 0:
                prompt_text += "\n\nREMINDER: Return ONLY a single valid JSON object. No code fences. No commentary."
            else:
                return prompt_text, 1

def print_slow(text, delay=0.05):
    """
    Prints text line by line with a delay.
    delay: seconds between each line
    """
    for line in text.splitlines():
        print(line)
        time.sleep(delay)

# if not api_key:
#     raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
# client = OpenAI(api_key=api_key)
def main():
    args = parse_args()
    input_file = args.input_file
    output_folder = args.output_dir

    # =============== CONFIG ===============
    MODEL = "o4-mini"
    USE_FULL_RUBRIC_IN_PROMPT = True  # set True if you want to inject full rubric JSON into the prompt
    INTENDED_AUDIENCE = "Physicians"
    USE_CASE = "Diagnosis Reasoning check"
    INPUT_NOTE_PATH = input_file
    OUTPUT_JSON_PATH = output_folder
    # =====================================

    # 1) Read inputs
    with open(INPUT_NOTE_PATH, "rb") as f:
        source_note = pickle.load(f)

    all_results = []
    all_errors = []
    for n in tqdm(source_note[:50]):
        rubric_block = ""
        if USE_FULL_RUBRIC_IN_PROMPT:
            rubric_block = "\n\nFULL_RUBRIC_JSON:\n" + json.dumps(PDSQI9_RUBRIC, ensure_ascii=False, indent=2)

        EVALUATOR_PROMPT = dedent(f"""
        {CONDENSED_RUBRIC_TEXT}{rubric_block}

        INTENDED_AUDIENCE: {INTENDED_AUDIENCE}
        USE_CASE: {USE_CASE}

        PROGRESS_NOTES:
        <<<
        {n['progress_notes']}
        >>>

        DIAGNOSIS: 
        <<<
        {n['diagnosis']}
        >>>

        LLM_DIAGNOSIS_REASONING:
        <<<
        {n['llm_reasoning']}
        >>>

        REQUIRED JSON SCHEMA:
        {{
        "citations": {{"score": int, "rationale": str}},
        "accuracy_extractive": {{
            "score": int,
            "rationale": str,
            "fabrication_examples": [{{"reasoning_text": str}}],
            "falsification_examples": [{{"reasoning_text": str, "correct_evidence": str}}]
        }},
        "thoroughness": {{
            "score": int,
            "rationale": str,
            "omissions": {{
            "pertinent": [str],
            "potentially_pertinent": [str]
            }}
        }},
        "usefulness": {{"score": int, "rationale": str}},
        "organization": {{"score": int, "rationale": str}},
        "comprehensibility": {{"score": int, "rationale": str}},
        "succinctness": {{"score": int, "rationale": str}},
        "synthesis_abstraction": {{"score": "N/A" | int, "rationale": str}},
        "stigmatizing_language": {{
            "present_in_progress_note": bool,
            "present_in_reasoning": bool,
            "examples": [str]
        }},
        "evidence_spans": [{{"reasoning_claim": str, "source_quote": str}}],
        "overall": {{
            "safe_for_use": bool,
            "justification": str
        }}
        }}
        """).strip()

    # 3) Call OpenAI in JSON mode, with a retry if JSON parsing fails
        result, error_code = run_eval(EVALUATOR_PROMPT, MODEL)
        if error_code == 0:
            all_results.append(result)
        else:
            all_errors.append({
                "progress_notes": n['progress_notes'],
                "diagnosis": n['diagnosis'],
                "llm_reasoning": n['llm_reasoning'],
                "prompt_text": EVALUATOR_PROMPT
            })

    # 4) Save + pretty print
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(args.error_path, "wb") as f:
        pickle.dump(all_errors, f)

    print(f"PDSQI-9 evaluation saved to {OUTPUT_JSON_PATH}\n")
    #print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
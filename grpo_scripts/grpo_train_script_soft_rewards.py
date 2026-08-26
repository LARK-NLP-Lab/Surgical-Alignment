import spacy
import yaml
import time
import json
import pickle
from collections import defaultdict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, EarlyStoppingCallback
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
# from sentence_transformers import SentenceTransformer
import re
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
import math
from collections import Counter
import ast
import gc
# import util
import evaluate
import random
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import os
from collections import deque, defaultdict
from transformers import BitsAndBytesConfig, TrainerCallback
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, PeftModel
import argparse
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint
import time
from dotenv import load_dotenv
from peft.utils import prepare_model_for_kbit_training
from quickumls import *
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

load_dotenv()

login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

accepted_semtypes=['T033','T040', 'T063', 'T037', 'T060', 'T055', 'T017', 'T069', 'T122', 'T038', 'T044', 'T130', 'T057', 'T073', 'T081', 'T185', 'T098', 'T101', 'T075', 'T184', 'T021', 'T200', 'T047', 'T022', 'T066', 'T068', 'T201', 'T089', 'T097', 'T023', 'T062', 'T070', 'T093', 'T190', 'T102', 'T001', 'T114', 'T041', 'T054', 'T049', 'T120', 'T169', 'T028', 'T131', 'T045', 'T026', 'T129', 'T092', 'T056', 'T065', 'T196', 'T123', 'T031', 'T064', 'T121', 'T058', 'T034', 'T039', 'T074', 'T018', 'T019', 'T059', 'T043', 'T104', 'T197', 'T077', 'T061', 'T099', 'T029', 'T030', 'T095', 'T020', 'T082', 'T042', 'T086', 'T032', 'T091', 'T083', 'T109', 'T046', 'T078', 'T072', 'T067', 'T090', 'T116']
matcher = QuickUMLS("../quickumls/",overlapping_criteria="score",similarity_name="cosine",threshold=0.90,accepted_semtypes=accepted_semtypes)

cui_to_entity_path = "/data/yifu/data/entity_linking/cui_to_entity.pkl" 
with open(cui_to_entity_path, 'rb') as handle:
    cui_to_entity = pickle.load(handle)

primekg = pd.read_csv('./primeKG/kg.csv', low_memory=False)

node_types = ['effect/phenotype', 'disease', 'exposure']
node_types2 = ['biological_process', 'anatomy']

def build_simstring_db(df, id_col="x_id", term_cols=("x_name", "x_source"), node_types=node_types):
    """
    Build a SimString DB from multiple dataframe columns.
    Returns a dict mapping normalized term -> (id, original_term).
    """
    db = DictDatabase(CharacterNgramFeatureExtractor(2))
    term_to_info = {}
    
    for _, row in df[(df['x_type'] == 'effect/phenotype') | (df['x_type'] == 'disease')].iterrows():
        for col in term_cols:
            val = row[col]
            if pd.notna(val):  # skip NaNs
                norm_val = val.lower()
                db.add(norm_val)
                term_to_info[norm_val] = (row[id_col], val)
    
    return db, term_to_info

def extract_matches(paragraph, db, term_to_info, threshold=0.8):
    """
    Extract approximate matches from paragraph against multiple dataframe columns.
    Returns list of dicts with id, matched_term, and matched_phrase.
    """
    searcher = Searcher(db, CosineMeasure())
    
    # Tokenize into words
    words = re.findall(r"\w+", paragraph.lower())
    
    # # Generate n-grams (1 up to max_ngram)
    # ngrams = [' '.join(words[i:i+n]) for n in range(1, max_ngram+1) 
    #           for i in range(len(words)-n+1)]
    
    results = []
    for word in words:
        # print("word is", word)
        # matches = list(db.retrieve(phrase))
        matches = searcher.search(word, threshold)
        if matches:
            for match in matches:
                id_val, orig_term = term_to_info[match]
                results.append({
                    "id": id_val,
                    "matched_term": orig_term,
                    "matched_phrase": word
                })
    return results

db, term_to_info = build_simstring_db(primekg)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="./models/Qwen7B_SFT_d1/", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/GRPO/Qwen7B_GRPO_d1', help='output directory')
    parser.add_argument('--input_dataset', type=str, default='./grpo_data/grpo_train_d1_selected_all_paths.pkl', help='dataset file')
    parser.add_argument('--val_dataset', type=str, default='./grpo_data/grpo_train_val_d1_selected_all_paths.pkl', help='val dataset file')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate')
    parser.add_argument("--task", type=str, default="d1")
    parser.add_argument('--is_prime', type=float, default=0, help='is it primeKG training')
    parser.add_argument('--is_resume', type=int, default=0, help='should we resume')
    args = parser.parse_args()
    return args

def formatting_prompts(textData, is_prime):
    return {"prompt": f"{textData['prompt']}", "prime": str(is_prime), "answer": f"{textData['answer']}", "label_path": textData['label_path'] if 'label_path' in textData else textData['answer'], "source": textData['source'], "task": textData["task"], "prediction_type": textData['prediction_type'] if 'prediction_type' in textData else None, "wrongOptions": textData['wrong_options'] if 'wrong_options' in textData else None}

def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, trust_remote_code=True)

    model_nf4.config.use_cache = False

    return tokenizer, model_nf4

def peft_model(model, model_name, is_pretrained):
    if is_pretrained:
        model = PeftModel.from_pretrained(model=model, model_id=model_name, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", 'o_proj'],
            bias="none",
        )

        model = get_peft_model(model, peft_config)
    return model

def print_trainable_params(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Reward for NHP
# We first apply quickUMLS matcher to the extracted model completion and the label answer.
# If we don't hit any matches, we go through the cui_to_entity.pkl dictionary and look for matches.
# Finally, we compare the CUIs for both.
def reward_fn_quickUMLS(extracted_ans, answer):
    pattern = r'<Answer>.*</Answer>'

    def find_entity(entity):
        entity_findings = []

        for k in cui_to_entity:
            for t in cui_to_entity[k]:
                if entity == t.capitalize():
                    entity_findings.append(k)

        return entity_findings
    
    if answer[0] == "|":
        extracted_cui = matcher.match(extracted_ans[1:] if len(extracted_ans) >= 1 and extracted_ans[0] == "|" else extracted_ans, )
        gold_cui = matcher.match(answer[1:], )

        if len(extracted_cui):
            extracted_cui = extracted_cui[0][0]['cui']
        else:
            extracted_cui = find_entity(extracted_ans[1:] if len(extracted_ans) >= 1 and extracted_ans[0] == "|" else extracted_ans)

            if len(extracted_cui):
                extracted_cui = extracted_cui[0]
            else:
                return 0
        
        if len(gold_cui):
            gold_cui = gold_cui[0][0]['cui']
        else:
            gold_cui = find_entity(answer[1:])[0]
        return extracted_cui == gold_cui
    else:
        return extracted_ans == answer

# Reward for NHP
# Similar to UMLS, just fit for PrimeKG
def reward_fn_primeKG(extracted_ans, answer):
    # print("extracted_ans", extracted_ans)
    # print("ans", answer)
    if answer[0] == "|":
        extracted_cui = extract_matches(extracted_ans[1:] if len(extracted_ans) >= 1 and extracted_ans[0] == "|" else extracted_ans, db, term_to_info)
        gold_cui = extract_matches(answer[1:], db, term_to_info)

        extracted_cui_ids = list(map(lambda x: x['id'], extracted_cui))
        gold_cui_ids = list(map(lambda x: x['id'], gold_cui))

        # print("gold_cui", gold_cui_ids)
        # print("extracted_cui_ids", extracted_cui_ids)
        if len(set(extracted_cui_ids).intersection(set(gold_cui_ids))):
            return 1
        else:
            return 0
    else:
        return extracted_ans == answer or extracted_ans == answer[2:]

def reward_func(**kwargs):

    rewards = []
    # pattern = r'<Answer>.*</Answer>'
    is_prime = kwargs['prime'][0]
    # print("keys", kwargs.keys())
    
    for completion, answer, task, wrongOptions, ans in zip(kwargs['completions'], kwargs['label_path'], kwargs['task'], kwargs['wrongOptions'], kwargs['answer']):

        # extracted_ans = re.findall(pattern, completion)
        # print("completion\n", completion.split("<Answer>"))
        if len(completion.split("<Answer>")) > 1 and len(completion.split("<Answer>")[1].split("</Answer>")) > 1:
            extracted_ans = f"<Answer>{completion.split("<Answer>")[1].split("</Answer>")[0]}</Answer>"
        else:
            rewards.append(-1)
            continue
        
        # print("=====PROMPT=====", kwargs['prompt'][0])
        # print("Completion\n", completion)
        # print("extracted_ans\n", extracted_ans)
        # print("actual ans\n", answer)

        if task == "Multi_Path_Selection": # Covers the case of PN@10
            # We check that the label path should be part of the model completion, and none of the wrong options should be part of the model's completion
            curr_reward = 0.1

            if isinstance(answer, list):
                for p in answer:
                    if p not in extracted_ans:
                        curr_reward = -1
                        break
                
                for p in wrongOptions:
                    if p in extracted_ans:
                        curr_reward = -1
                        break
                
        elif task == "Path_Selection": # Covers cases of P@10 and P@2
            # We check that the label path should be part of the model completion, and none of the wrong options should be part of the model's completion
            curr_reward = 0.1

            # print("extracted_ans\n", extracted_ans)
            # print("actual ans\n", answer)

            if isinstance(answer, list):
                for p in answer:
                    if p not in extracted_ans:
                        curr_reward = -1
                        break
            else:
                if answer not in extracted_ans:
                    curr_reward = -1

            if isinstance(wrongOptions, list): 
                for p in wrongOptions:
                    if p in extracted_ans:
                        curr_reward = -1
                        break
            else:
                if wrongOptions in extracted_ans:
                    curr_reward = -1

        elif task == "NHP": # Covers NHP case
            if is_prime:
                curr_reward = reward_fn_primeKG(extracted_ans.strip(), ans)
            else:
                curr_reward = reward_fn_quickUMLS(extracted_ans.strip(), ans)

        else: # PC case left
            # Simply matching the ground truth label with the model completion
            curr_reward = 0.1

            if ans not in extracted_ans:
                curr_reward = -1
        
        # if curr_reward > 0:
        #     print("Reward + 1 for this case")
        rewards.append(curr_reward)
        
    # if sum(rewards) > 0:
    #     print("some rewards +1 !")
    return rewards

accepted_semtypes=['T033','T040', 'T063', 'T037', 'T060', 'T055', 'T017', 'T069', 'T122', 'T038', 'T044', 'T130', 'T057', 'T073', 'T081', 'T185', 'T098', 'T101', 'T075', 'T184', 'T021', 'T200', 'T047', 'T022', 'T066', 'T068', 'T201', 'T089', 'T097', 'T023', 'T062', 'T070', 'T093', 'T190', 'T102', 'T001', 'T114', 'T041', 'T054', 'T049', 'T120', 'T169', 'T028', 'T131', 'T045', 'T026', 'T129', 'T092', 'T056', 'T065', 'T196', 'T123', 'T031', 'T064', 'T121', 'T058', 'T034', 'T039', 'T074', 'T018', 'T019', 'T059', 'T043', 'T104', 'T197', 'T077', 'T061', 'T099', 'T029', 'T030', 'T095', 'T020', 'T082', 'T042', 'T086', 'T032', 'T091', 'T083', 'T109', 'T046', 'T078', 'T072', 'T067', 'T090', 'T116']
matcher = QuickUMLS("../quickumls/",overlapping_criteria="score",similarity_name="cosine",threshold=0.90,accepted_semtypes=accepted_semtypes)

def get_path_labels(prediction, label):
    
    # for d in diagnoses.split(";"):
    output = [(ii['ngram'],ii['term'],ii['cui'], ii['semtypes']) for i in matcher.match(prediction,) for ii in i]
    output_label = [(ii['ngram'],ii['term'],ii['cui'], ii['semtypes']) for i in matcher.match(label,) for ii in i]

    output_cuis = set()
    output_label_cuis = set()

    for m in output:
        output_cuis.add(m[2])

    for m in output_label:
        output_label_cuis.add(m[2])

    return output_cuis, output_label_cuis

def reward_cui_f_answer(**kwargs):
    rewards = []
    # pattern = r'<Answer>.*</Answer>'
    is_prime = kwargs['prime'][0]
    # print("keys", kwargs.keys())
    
    for completion, answer, task, wrongOptions, ans in zip(kwargs['completions'], kwargs['label_path'], kwargs['task'], kwargs['wrongOptions'], kwargs['answer']):

        # extracted_ans = re.findall(pattern, completion)
        # print("completion\n", completion.split("<Answer>"))

        print("predicted answer", completion)
        print("gold", ans)
        if len(completion.split("<Answer>")) > 1 and len(completion.split("<Answer>")[1].split("</Answer>")) > 1:
            extracted_ans = f"<Answer>{completion.split("<Answer>")[1].split("</Answer>")[0]}</Answer>"
        else:
            extracted_ans = completion

        extracted_ans = extracted_ans.replace("|", " ")
        extracted_ans = extracted_ans.replace("->", " ")

        extracted_gold_ans = ans.replace("|", " ")
        extracted_gold_ans = extracted_gold_ans.replace("->", " ")

        notes_cui, gold_cui = get_path_labels(extracted_ans, extracted_gold_ans)
        print("notes_cui", notes_cui)
        print("gold_cui", gold_cui)

        if len(gold_cui) == 0:
            # continue
            print("invalid case !: gold cui 0")

        if len(notes_cui) == 0:
            # num_invalid += 1
            # accs.append(0)
            # prec_list.append(0)
            # recall_list.append(0)
            print("invalid case !: notes cui 0")
        else:
            prec = len(gold_cui.intersection(notes_cui)) / len(notes_cui) 
            rec = len(gold_cui.intersection(notes_cui)) / len(gold_cui) 

            if prec + rec == 0:
                # accs.append(0)
                # prec_list.append(0)
                # recall_list.append(0)
                print("Prec + recall 0!")
                rewards.append(0)
            else:
                acc = 2*(prec*rec) / (prec+rec)

                # accs.append(acc)
                # prec_list.append(prec)
                # recall_list.append(rec)
                print("f score ", acc)
                rewards.append(acc)

        # if curr_reward > 0:
        #     print("Reward + 1 for this case")
        # rewards.append(curr_reward)
        
    if sum(rewards) > 0:
        print("some rewards +1 !")
    return rewards 

def reward_partial(**kwargs):
    rewards = []
    # pattern = r'<Answer>.*</Answer>'
    is_prime = kwargs['prime'][0]
    # print("keys", kwargs.keys())
    
    for completion, answer, task, wrongOptions, ans in zip(kwargs['completions'], kwargs['label_path'], kwargs['task'], kwargs['wrongOptions'], kwargs['answer']):

        # extracted_ans = re.findall(pattern, completion)
        # print("completion\n", completion.split("<Answer>"))

        # print("predicted answer", completion)
        # print("gold", ans)
        if len(completion.split("<Answer>")) > 1 and len(completion.split("<Answer>")[1].split("</Answer>")) > 1:
            extracted_ans = f"<Answer>{completion.split("<Answer>")[1].split("</Answer>")[0]}</Answer>"
        else:
            extracted_ans = completion
        

        extracted_ans = extracted_ans.replace("|", " ")
        extracted_ans = extracted_ans.replace("->", " ")

        extracted_gold_ans = ans.replace("|", " ")
        extracted_gold_ans = extracted_gold_ans.replace("->", " ")

        reward = path_alignment_reward_func(extracted_ans, extracted_gold_ans)

        # print("reward", reward)
        rewards.append(reward)

    return rewards

class RewardEarlyStoppingCallback(TrainerCallback):
    def __init__(self, tokenizer, metric_key="eval_reward", patience=3, greater_is_better=True):
        super().__init__()
        self.metric_key = metric_key
        self.patience = patience
        self.best = -math.inf if greater_is_better else math.inf
        self.greater_is_better = True
        self.tokenizer = tokenizer
    
    def _get_metric(self, state, metrics):
        for entry in reversed(state.log_history):
            if self.metric_key in entry:
                current = entry[self.metric_key]
                return current
        return None

    def on_evaluate(self, args, state, control, model, **kwargs):
        current = None

        for entry in reversed(state.log_history):
            if self.metric_key in entry:
                current = entry[self.metric_key]
                break
        
        if current is None:
            return control
    
        improved = (current > self.best) if self.greater_is_better else (current < self.best)

        if improved:
            self.best = current
            self.num_bad = 0
            control.should_save = True
            self.best_checkpoint = f"{args.output_dir}/best_checkpoint"
            model.save_pretrained(self.best_checkpoint)
            self.tokenizer.save_pretrained(self.best_checkpoint)
            print(f"New best: {self.best:.4f} - checkpoint saved to {self.best_checkpoint}")
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                control.should_training_stop = True
                control.should_save = False
        
        print("The number of bad evals", self.num_bad)
        return control


# =====================================================================
#                       Reward Functions
# =====================================================================

STOP_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'can', 'this', 'that', 'these', 'those'
}


def normalize_tokens(text: str):
    """Normalize text to tokens, removing stop words."""
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t and t not in STOP_WORDS]

def repetition_penalty_factor(tokens, threshold: float = 0.35):
    """Calculate penalty for repetitive text."""
    if not tokens:
        return 1.0
    
    from collections import Counter
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    ratio = most_common / max(1, len(tokens))
    
    base = max(0.0, 1.0 - max(0.0, ratio - threshold) * 3.0)
    
    # Penalty for consecutive repeats
    max_run = 1
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    
    run_penalty = 1.0 - max(0.0, (max_run - 3)) * 0.05
    return max(0.0, base * run_penalty)

def path_alignment_reward_func(completion, answer):
    """
    Reward function for knowledge graph path alignment.
    Measures overlap between model reasoning and KG paths.
    """
    # Build token sets
    path_tokens = set(normalize_tokens(answer))
    thinking_tokens_list = normalize_tokens(completion)
    thinking_tokens_set = set(thinking_tokens_list)
    
    if not path_tokens:
        return 0.0
    
    # Calculate overlap
    hits = thinking_tokens_set & path_tokens
    coverage = len(hits) / max(1, len(path_tokens))
    min_unique_hit = 1.0 if len(hits) >= 2 else 0.0
    
    # Apply repetition penalty
    rep_factor = repetition_penalty_factor(thinking_tokens_list)
    
    base_reward = (1.2 * coverage + 0.3 * min_unique_hit)
    return min(base_reward * rep_factor, 1.5)


def main():
    start_time = time.time()

    args = parse_args()
    model_name = args.model_name
    output_folder = args.output_dir
    input_dataset = args.input_dataset
    val_dataset = args.val_dataset
    learning_rate = args.learning_rate
    is_prime = args.is_prime == 1
    resume = args.is_resume
    print("is_prime", is_prime)
    os.makedirs(output_folder, exist_ok=True)

    is_pretrained = False

    if os.path.exists(model_name):
        model_name = get_last_checkpoint(model_name)
        print("the checkpoint fetched is", model_name)
        is_pretrained = True

    tokenizer, model = get_quantized_models(model_name)
    # model = prepare_model_for_kbit_training(model)
    # model.get_input_embeddings().weight.requires_grad_(False)
    model = peft_model(model, model_name, is_pretrained)
    # if os.path.exists(model_name):
    #     model = PeftModel.from_pretrained(model, model_name)

    print_trainable_params(model)
    with open(input_dataset, 'rb') as f: 
        prompt_data_train = pickle.load(f)

    with open(val_dataset, 'rb') as f: 
        prompt_data_val = pickle.load(f)

    datasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x, is_prime), prompt_data_train)))
    evalDatasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x, is_prime), prompt_data_val[:250])))

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = GRPOConfig(
        output_dir=f"./{output_folder}", 
        per_device_train_batch_size=6,
        learning_rate=learning_rate, 
        num_train_epochs=1, 
        per_device_eval_batch_size=6,
        beta=0.001,
        logging_dir= f'./{output_folder}/logs',
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,

        logging_steps=25,
        dataloader_pin_memory=True,
        save_total_limit=2,
        gradient_accumulation_steps=12,
        num_generations=6,

        bf16=True,
        use_vllm=True,
        vllm_mode="colocate"
    )

    trainer = GRPOTrainer(
        model,
        args=sft_config,
        train_dataset=datasetFromList,
        eval_dataset=evalDatasetFromList,
        reward_funcs=[reward_func, reward_partial],
        # reward_funcs=[reward_partial],
        callbacks=[RewardEarlyStoppingCallback(tokenizer, patience=2)]
    )

    if resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric:", trainer.state.best_metric)

    df = pd.DataFrame(trainer.state.log_history)

    df.to_csv(f'./{output_folder}/scores.csv', index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == '__main__':
    main()
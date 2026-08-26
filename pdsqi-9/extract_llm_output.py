import spacy
import yaml
import time
import json
import pickle
from collections import defaultdict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
# from sentence_transformers import SentenceTransformer
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math
from collections import Counter
import ast
import gc
# import util
import evaluate
import random
from transformers import BartForConditionalGeneration
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import os
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel
import argparse
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))
torch.set_default_device("cuda")
def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, device_map="auto")

    return tokenizer, model_nf4

def peft_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", 'gate_proj'],
        # bias="none",
    )

    model = get_peft_model(model, peft_config)

    return model

def peft_model_pretrained(model, model_name):
    model = PeftModel.from_pretrained(model=model, model_id=model_name)

    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/Qwen7B_SFT_d1', help='output directory')
    args = parser.parse_args()
    return args

def main():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    args = parse_args()
    test_df="../../BioNLP2023-1A-Test.csv"
    data = pd.read_csv(test_df)
    
    sb=data["Subjectives"].tolist()
    ob=data["Objectives"].tolist()
    ass=data["Assessment"].tolist()
    sm=data["Summary"].tolist()
    f_id=data["File ID"].tolist()

    note = []
    for i in range(len(sm)):
        if type(sb[i]) == str and type(ob[i]) == str and type(ass[i]) == str and type(sm[i]) == str:
            note.append((sb[i]+"\n"+ob[i]+'\n'+ass[i],sm[i], f_id[i], i))

    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config).to("cuda:0")
    
    if os.path.exists(model_name):
        model = PeftModel.from_pretrained(model, model_name)
    
    all_responses = []
    
    for n in tqdm(note[:50]):
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES of a patient.
        You must review the PROGRESS_NOTES and provide the diagnosis for the patient. Explain your reasoning thoroughly for each diagnosis.'''

        prompt = PROMPT_CONS + f"\n\nPROGRESS_NOTES: {n[0]}" + "\n\nPlease predict the diagnoses that the patient might have.\nOutput:"
        model_inputs = tokenizer(prompt, return_tensors="pt")
        generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # print("===ground diagnoses====")
        # print(n[1])
        # print("=====output=====")
        # print(output.split("\nOutput:")[1])

        temp = {
            'progress_notes': n[0],
            'diagnosis': n[1],
            'llm_reasoning': output.split("\nOutput:")[1]
        }

        all_responses.append(temp)

    # with open("temp2.pkl", "wb") as f:
    #     pickle.dump(all_re sponses, f)
    
    with open(args.output_dir, "wb") as f:
        pickle.dump(all_responses, f)
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == '__main__':
    main()
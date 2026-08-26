
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
# from sentence_transformers import SentenceTransformer
import re
import numpy as np
import torch
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field
from collections import Counter
from transformers import Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, DataCollatorForSeq2Seq, DataCollatorWithPadding
import pandas as pd
from tqdm.notebook import tqdm
import os
from peft import get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType
from transformers.trainer_utils import get_last_checkpoint
from transformers import pipeline
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record
from accelerate import Accelerator
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
import pickle
from torch.nn import KLDivLoss
from KLDiv_dataset import KLDivDataset
import random
from torch.utils.data import Subset

load_dotenv()

# torch.set_default_device('cuda')

login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

seed = 42
torch.manual_seed(seed)
random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='model name')
    parser.add_argument('--output_dir', type=str, default='./hyperparam_search/Qwen7B_SFT_d1_temp', help='output directory')
    parser.add_argument('--input_dataset', type=str, default='../final_training_data/train_d1.pkl', help='dataset file')
    parser.add_argument('--val_dataset', type=str, default='../final_training_data/train_val_d1.pkl', help='val dataset file')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
    args = parser.parse_args()
    return args

def formatting_prompts(textData):
    return {"prompt": f"{textData['prompt']}", "completion": f"{textData['answer']}"}

def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config)

    return tokenizer, model_nf4


def peft_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", 'gate_proj'],
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    return model

@dataclass
class KLDivConfig(SFTConfig):
    """
    Extends SFTConfig to include hyperparameter tuning specific args.
    """
    lambda_kl: float = field(
        default=0.05,
        metadata={"help": "The coefficient for the KL Divergence loss term."}
    )

class KLDivSFTTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kl_fn = KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.lambda_kl = self.args.lambda_kl

        sft_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        logits = outputs.logits

        # Compute reference logits by disabling the adapter
        with torch.no_grad(), model.disable_adapter():
            ref_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
        ref_logits = ref_outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        base_probs = torch.softmax(ref_logits, dim=-1)

        kl_loss = self.kl_fn(log_probs, base_probs)

        loss = sft_loss + self.lambda_kl * kl_loss

        return (loss, outputs) if return_outputs else loss


def optuna_hp_space(trial):
    return {
        "lambda_kl": trial.suggest_float("lambda_kl", 0.01, 1, log=True)
        # "learning_rate": trial.suggest_float('learning_rate', 3e-5, 3e-5)
    }

# --- 4. Compute Objective for Optuna ---
def compute_objective(metrics):
    """
    Extracts the objective metric (eval_loss) from the metrics dictionary.
    Returns float('inf') if missing to prevent Optuna crashes.
    """
    print("The metrics provided are", metrics.keys())
    if metrics is None:
        return float("inf")
    
    # Check for standard eval_loss
    loss = metrics.get("eval_loss")
    
    # Fallback: sometimes prefixes appear or loss is missing
    if loss is None:
        # Check for 'loss' (unlikely in eval dict but possible)
        loss = metrics.get("loss")
        
    if loss is None:
        print(f"WARNING: eval_loss not found in metrics keys: {metrics.keys()}. Returning inf.")
        return float("inf")
        
    return float(loss)

def main():
    args = parse_args()
    model_name = args.model_name
    output_folder = args.output_dir
    input_dataset = args.input_dataset
    val_dataset = args.val_dataset
    learning_rate = args.learning_rate


    os.makedirs(output_folder, exist_ok=True)

    def model_init(trial):
        # Define your model instantiation logic here
        # Access hyperparameters using trial.suggest_float, trial.suggest_int, etc.
        # learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        # ... other custom hyperparameters
        tokenizer, model = get_quantized_models(model_name)
        # Instantiate your Hugging Face model and configure it with the suggested hyperparameters
        model = peft_model(model)
        # You might need to modify the model's configuration based on custom hyperparameters
        return model

    with open(input_dataset, 'rb') as f: 
        prompt_data_train = pickle.load(f)

    with open(val_dataset, 'rb') as f: 
        prompt_data_val = pickle.load(f)    
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    # 500 - 1000
    train_dataset = KLDivDataset(list(map(lambda x: formatting_prompts(x), prompt_data_train[:600])), tokenizer)
    train_val_dataset = KLDivDataset(list(map(lambda x: formatting_prompts(x), prompt_data_val[:200])), tokenizer)

    sft_config = KLDivConfig(
        output_dir=f"./{output_folder}", 
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,
        packing=True, 
        learning_rate=3e-5, 
        num_train_epochs=1,
        logging_dir= f'./{output_folder}/logs',
        eval_strategy="steps",
        eval_steps=200,
        logging_strategy="steps",
        logging_steps=200,
        save_strategy="no",
        # label_names=["completion"],
        lambda_kl=0.05, # will be overriden by optuna
        prediction_loss_only=True
    )

    trainer = KLDivSFTTrainer(
        model_init=model_init,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=train_val_dataset,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
    )

    best_run = trainer.hyperparameter_search(
        direction=["minimize"],
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=10,
        compute_objective=compute_objective,
        study_name=f"{output_folder.split("/")[-1]}",
        storage=f"sqlite:///{output_folder}/optuna_trials.db"        
    )

    print("the best hyperparameters are", best_run.hyperparameters)


if __name__ == '__main__':
    main()
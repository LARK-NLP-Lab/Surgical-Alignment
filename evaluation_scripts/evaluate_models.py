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
from transformers import BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
from peft import LoraConfig, TaskType, PeftModel
import argparse
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers import pipeline
from tqdm import tqdm
from quickumls import *
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

torch.set_default_device("cuda")

def get_data_label(data, id):
    answer = set()
    for d in data:
        if d['id'] == id:
            answer.add(d['label_path'])
    return answer

def get_path_labels(prediction, label):
    accepted_semtypes=['T033','T040', 'T063', 'T037', 'T060', 'T055', 'T017', 'T069', 'T122', 'T038', 'T044', 'T130', 'T057', 'T073', 'T081', 'T185', 'T098', 'T101', 'T075', 'T184', 'T021', 'T200', 'T047', 'T022', 'T066', 'T068', 'T201', 'T089', 'T097', 'T023', 'T062', 'T070', 'T093', 'T190', 'T102', 'T001', 'T114', 'T041', 'T054', 'T049', 'T120', 'T169', 'T028', 'T131', 'T045', 'T026', 'T129', 'T092', 'T056', 'T065', 'T196', 'T123', 'T031', 'T064', 'T121', 'T058', 'T034', 'T039', 'T074', 'T018', 'T019', 'T059', 'T043', 'T104', 'T197', 'T077', 'T061', 'T099', 'T029', 'T030', 'T095', 'T020', 'T082', 'T042', 'T086', 'T032', 'T091', 'T083', 'T109', 'T046', 'T078', 'T072', 'T067', 'T090', 'T116']
    matcher = QuickUMLS("../quickumls/",overlapping_criteria="score",similarity_name="cosine",threshold=0.90,accepted_semtypes=accepted_semtypes)
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

def get_prediction_labels(prediction, label):
    accepted_semtypes=['T033', 'T037', 'T046', 'T047', 'T048', 'T049', 'T184']
    matcher = QuickUMLS("../quickumls/",overlapping_criteria="score",similarity_name="cosine",threshold=0.90,accepted_semtypes=accepted_semtypes)

    output = [(ii['ngram'],ii['term'],ii['cui'], ii['semtypes']) for i in matcher.match(prediction,) for ii in i]
    output_label = [(ii['ngram'],ii['term'],ii['cui'], ii['semtypes']) for i in matcher.match(label,) for ii in i]

    output_cuis = set()
    output_label_cuis = set()

    for m in output:
        output_cuis.add(m[2])

    for m in output_label:
        output_label_cuis.add(m[2])

    return output_cuis, output_label_cuis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', type=str, default="./Qwen1.5B/", help='model name')
    parser.add_argument('--data_file', type=str, default="sft_val_data.pkl", help='test data')
    parser.add_argument('--eval_metric', type=str, default="d1", help='which metric to run')
    parser.add_argument('--with_diagnosis', type=int, default=0, help="if diagnosis should be provided in the prompt")
    args = parser.parse_args()
    return args

def get_multiple_choice_answer(model, tokenizer, data_file, with_diagnosis):
    if with_diagnosis:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES and the DIAGNOSES AND FINDINGS of a patient.
You are provided with 4 knowledge base paths. You must select the Path which leads to a valid diagnosis.

The Paths are provided as :
A. PATH 1
B. PATH 2
C. PATH 3
D. PATH 4

Each path follows the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.

Your answer needs to be a single word with your pick for the right answer.

For eg. if the options are the following:

A. PATH 1
B. PATH 2
C. PATH 3
D. PATH 4

If you think that "PATH 2" leads to a correct diagnosis according to the provided patient records, then your output must be: "B".
Your Answer CANNOT be anythin other than "A" or "B" or "C" or "D".
'''

    else:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES and the DIAGNOSES AND FINDINGS of a patient.
You are provided with 4 knowledge base paths. You must select the Path which leads to a valid diagnosis.

The Paths are provided as :
A. PATH 1
B. PATH 2
C. PATH 3
D. PATH 4

Each path follows the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.

Your answer needs to be a single word with your pick for the right answer.

For eg. if the options are the following:

A. PATH 1
B. PATH 2
C. PATH 3
D. PATH 4

If you think that "PATH 2" leads to a correct diagnosis according to the provided patient records, then your output must be: "B".
Your Answer CANNOT be anything other than "A" or "B" or "C" or "D".
'''

    rouge = evaluate.load('rouge')
    answers = []
    num_correct_picks = 0
    # flag = 1
    for data in tqdm(data_file[:5]):
        prompt = PROMPT_CONS + f"\n\nPROGRESS_NOTES: {data['progress_note']}"
        answer = data['label_path'] if 'label_path' in data.keys() else data['answer']

        all_answers = [answer]
        all_answers.extend(random.sample(data['wrong_options'], 3))
        
        if with_diagnosis:
            note = openTrainFile()
            labels_note = note[data['id']][1].split(';')
            prompt += f"\nDIAGNOSES AND FINDINGS: {','.join(labels_note)}"

        prompt += f"\nCANDIDATE_PATHS:\n"
        random.shuffle(all_answers)

        answerPrompt = ""

        prompt += f"A. {all_answers[0]}\n"
        prompt += f"B. {all_answers[1]}\n"
        prompt += f"C. {all_answers[2]}\n"
        prompt += f"D. {all_answers[3]}\n"

        answerPrompt += f"A. {all_answers[0]}\n"
        answerPrompt += f"B. {all_answers[1]}\n"
        answerPrompt += f"C. {all_answers[2]}\n"
        answerPrompt += f"D. {all_answers[3]}\n"
        
        answerOption = "A"

        if all_answers[1] == answer:
            answerOption = "B"
        elif all_answers[2] == answer:
            answerOption = "C"
        elif all_answers[3] == answer:
            answerOption = "D"

        prompt += '''\nBased on the PROGRESS_NOTES, which option can lead to valid diagnosis ?\nOutput [A or B or C or D]:'''
        
        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=1)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = output.split("\nOutput [A or B or C or D]:")[1]
            answers.append({
                'actual': answerOption,
                'prediction': pred
            })
            
            # m = rouge.compute(predictions=[pred], references=[data['label_path'] if 'label_path' in data.keys() else data['answer']])['rougeL']
            # flag = 0
            # for w in data['wrong_options']:
            #     if rouge.compute(predictions=[pred], references=[w])['rougeL'] > m:
            #         flag = 1
            #         break
            
            # if flag == 0:
            #     num_correct_picks += 1
            # print("all options", '\n'.join(all_answers))
            # print("prediction", pred)
            # print("answerOption", answerOption)
            if answerOption.capitalize() == pred.strip().capitalize():
                num_correct_picks += 1
            
    return num_correct_picks / len(data_file)

def get_next_hop_predictions(model, tokenizer, data_file):
    PROMPT_CONS = '''Review the PROGRESS_NOTES of a patient.\n'''

    answers = []

    for data in tqdm(data_file):
        # print("data", data)
        prompt = PROMPT_CONS + f"\n\nPROGRESS_NOTES: {data['progress_note']}"
        prompt += f"\nPATH: {'->'.join(data['answer'].split('->')[:2])}\nWhat would be the next step for the above PATH ?\nAnswer:"

        # prompt = data['prompt']
        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            generated_ids = model.generate(**model_inputs)

            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("output is", output)
            # answers.append({
            #     'actual': data['answer'],
            #     'prediction': output
            # })
            
    return answers

def openTrainFile():
    train_df="../BioNLP2023-1A-Train.csv"
    data = pd.read_csv(train_df)
    # print(data.head(3))
    # print(len(data))
    sb=data["Subjective Sections"].tolist()
    ob=data["Objective Sections"].tolist()
    ass=data["Assessment"].tolist()
    sm=data["Summary"].tolist()
    f_id=data["File ID"].tolist()

    note = []
    for i in range(len(sm)):
        if type(sb[i]) == str and type(ob[i]) == str and type(ass[i]) == str and type(sm[i]) == str:
            note.append((sb[i]+"\n"+ob[i]+'\n'+ass[i],sm[i], f_id[i], i))
    
    return note


def get_predictions_completions(model, tokenizer, data_file, with_diagnosis):
    if with_diagnosis:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES and the DIAGNOSES AND FINDINGS of a patient.
        You must review the knowledge base paths in CANDIDATE_PATHS based on concepts from PROGRESS_NOTES and select ONE PATH which can lead to one of the provided DIAGNOSES and FINDINGS.
        Each path in CANDIDATE_PATHS follow the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
        The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
        '''
    else:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES of a patient.
        You must review the knowledge base paths in CANDIDATE_PATHS based on concepts from PROGRESS_NOTES and select ONE PATH which can lead to a correct diagnosis.
        Each path in CANDIDATE_PATHS follow the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
        The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
        '''
#     PROMPT_CONS += '''\n
# Your OUTPUT must be just the path which can lead to valid diagnosis according to you.
# '''

    rouge = evaluate.load('rouge')
    answers = []
    num_correct_picks = 0

    notes = openTrainFile()
    # flag = 1
    # data_file[10:15]
    accs = []
    prec_list= []
    recall_list = []
    
    for data in tqdm(data_file[:1000]):
        # data = data_file[num]
        # print("\nnew data\n")
        prompt = PROMPT_CONS + f"\nPROGRESS_NOTES: {data['progress_note']}"
        all_answers = [data['label_path'] if 'label_path' in data.keys() else data['answer']]
        all_answers.extend(data['wrong_options'])
        
        if with_diagnosis:
            # print('adding diagnosis', with_diagnosis)
            note = openTrainFile()
            labels_note = note[data['id']][1].split(';')
            prompt += f"\nDIAGNOSES AND FINDINGS: {','.join(labels_note)}"

        prompt += f"\nCANDIDATE_PATHS:\n"
        random.shuffle(all_answers)

        for ind in range(10):
            prompt += f"{ind + 1}. {all_answers[ind]}\n"
        
        prompt += '''\nBased on the PROGRESS_NOTES, which PATH among the CANDIDATE_PATHS which can lead to valid diagnosis ? Pick ONLY ONE PATH which you think is valid to reach a diagnosis and output the path that you have picked inside the <Answer>..</Answer> tag.
        
Put your reasoning inside <Reasoning>..</Reasoning> tags.

Your output should be of the format:
<Reasoning> Your Reasoning </Reasoning>
<Answer> The path you have picked </Answer> <|endoftext|>
Respond under 200 words.
Output:
'''
#         prompt += '''\nBased on the PROGRESS_NOTES, which path among the CANDIDATE_PATHS is most likely to reach a valid diagnosis ? Think step by step, Pick ONLY ONE PATH and print the whole path wrapped between the <Answer> and </Answer> tags.
# Respond under 200 words.
# Output:
# '''

        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            pred = output.split("\nOutput:")[1]
            # print("===PROMPT===")

            # print(prompt)

            # print("===Answer===")

            # print(output)

            # print("====Actual Answer====")

            # print(data['answer'])

            extracted_answer = None
            if len(pred.split("<Answer>")) > 1:
                extracted_answer = pred.split("<Answer>")[1].split("</Answer>")[0]

            # print("=====ALL Options=====")

            # print("\n".join(all_answers))

            # print("====extracted_answ====")
            # print(extracted_answer)

            answers.append({
                'actual': data['answer'],
                'prediction': pred if extracted_answer == None else extracted_answer
            })
            

            # print("ground diagnoses are", notes[data['id']][1])
            
            # m = rouge.compute(predictions=[pred], references=[data['label_path'] if 'label_path' in data.keys() else data['answer']])['rougeL']
            # flag = 0
            # for w in data['wrong_options']:
            #     if rouge.compute(predictions=[pred], references=[w])['rougeL'] > m:
            #         flag = 1
            #         break
            
            # if flag == 0:
            #     num_correct_picks += 1
            # print("all options", '\n'.join(all_answers))
            
            # print("\nanswer", data['answer'])
            # print("\nprediction", output.split('Answer:')[1])
            # print("\n")
            gold_answer = data['answer'].replace("|", " ")
            gold_answer = gold_answer.replace("->", " ")

            predicted = answers[-1]['prediction'].replace("<Answer>", " ")
            predicted = predicted.replace("</Answer>", " ")
            predicted = predicted.replace("|", " ")
            predicted = predicted.replace("->", " ")
            notes_cui, gold_cui = get_path_labels(predicted, gold_answer)

            if len(gold_cui) == 0:
                continue

            if len(notes_cui) == 0:
                # num_invalid += 1
                accs.append(0)
                prec_list.append(0)
                recall_list.append(0)
            else:
                prec = len(gold_cui.intersection(notes_cui)) / len(notes_cui) 
                rec = len(gold_cui.intersection(notes_cui)) / len(gold_cui) 

                if prec + rec == 0:
                    accs.append(0)
                    prec_list.append(0)
                    recall_list.append(0)
                else:
                    acc = 2*(prec*rec) / (prec+rec)

                    accs.append(acc)
                    prec_list.append(prec)
                    recall_list.append(rec)

    print("Mean np", np.mean(accs))
    print("Precision np", np.mean(prec_list))
    print("Recall np", np.mean(recall_list))
    return answers, num_correct_picks 

def calculate_multiple_paths(model, tokenizer, data_file, with_diagnosis):
    # print("\nNew Model\n")
    if with_diagnosis:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES and the DIAGNOSES AND FINDINGS of a patient.
You must review the knowledge base paths in CANDIDATE_PATHS based on PROGRESS_NOTES and select the PATHS which can lead to one of the provided DIAGNOSES and FINDINGS.
There can be multiple correct Paths in the CANDIDATE_PATHS, you must select all of them.
Each path in CANDIDATE_PATHS follow the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.'''
    else:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES.
You must review the knowledge base paths in CANDIDATE_PATHS based on PROGRESS_NOTES and select the PATHS which can lead to a valid diagnosis.
There can be multiple correct Paths in the CANDIDATE_PATHS, you must select all of them.
Each path in CANDIDATE_PATHS follow the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.'''

    rouge = evaluate.load('rouge')
    answers = []

    accs = []
    prec_list= []
    recall_list = []
    for data in tqdm(data_file[:1000]):
        # print("source", data['source'])
        # print("data keys", data.keys())
        prompt = PROMPT_CONS + f"\nPROGRESS_NOTES: {data['progress_note']}"
        all_answers = data['label_paths'] if 'label_paths' in data.keys() else data['label_path']
        all_answers.extend(data['wrong_options'])
        

        prompt += f"\nCANDIDATE_PATHS:\n"
        random.shuffle(all_answers)

        for ind in range(10):
            prompt += f"{ind + 1}. {all_answers[ind]}\n"
        
        prompt += f'''\nFrom the above mentioned information, pick all the valid paths from the CANDIDATE_PATHS that can reach valid diagnosis.
Output each path that you have picked as single lines inside the <Answer>...</Answer> tags. Dont repeat the paths.
Output:'''
        
        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=256)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = output.split("\nOutput:")[1]

            # print("=====ALL Options=====")

            # print("\n".join(all_answers))
            extracted_answer = None
            if len(pred.split("<Answer>")) > 1:
                extracted_answer = pred.split("<Answer>")[1].split("</Answer>")[0]

            # print("=====ALL Options=====")

            # print("\n".join(all_answers))

            # print("====all pred====")
            # print(pred)

            # print("====extracted_answ====")
            # print(extracted_answer)

            # print("=====answer=====")
            # print(data['answer'])
            answers.append({
                'actual': data['answer'],
                'prediction': pred if extracted_answer == None else extracted_answer
            })
            

            # print("all options", '\n'.join(all_answers))
            
            # print("\nanswer", data['answer'])
            # print("\nprediction", output.split('Answer:')[1])
            # print("\n")

            gold_answer = data['answer'].replace("|", " ")
            gold_answer = gold_answer.replace("->", " ")

            predicted = answers[-1]['prediction'].replace("<Answer>", " ")
            predicted = predicted.replace("</Answer>", " ")
            predicted = predicted.replace("|", " ")
            predicted = predicted.replace("->", " ")
            notes_cui, gold_cui = get_path_labels(predicted, gold_answer)

            # print("gold")
            # print(gold_cui)
            # print("extracted")
            # print(notes_cui)

            if len(gold_cui) == 0:
                continue

            if len(notes_cui) == 0:
                # num_invalid += 1
                accs.append(0)
                prec_list.append(0)
                recall_list.append(0)
            else:
                prec = len(gold_cui.intersection(notes_cui)) / len(notes_cui) 
                rec = len(gold_cui.intersection(notes_cui)) / len(gold_cui) 

                if prec + rec == 0:
                    accs.append(0)
                    prec_list.append(0)
                    recall_list.append(0)
                else:
                    acc = 2*(prec*rec) / (prec+rec)

                    accs.append(acc)
                    prec_list.append(prec)
                    recall_list.append(rec)
    print("Mean np", np.mean(accs))
    print("Precision np", np.mean(prec_list))
    print("Recall np", np.mean(recall_list))
    return rouge.compute(predictions=list(map(lambda x: x['prediction'], answers)),
                        references=list(map(lambda x: x['actual'], answers)))
    # return answers
    


def calculate_rouge(model, tokenizer, data_file, with_diagnosis):
    answers, num_correct_picks = get_predictions_completions(model, tokenizer, data_file, with_diagnosis)
    # print("answer", answers)
    rouge = evaluate.load('rouge')

    return rouge.compute(predictions=list(map(lambda x: x['prediction'], answers)),
                        references=list(map(lambda x: x['actual'], answers))), num_correct_picks / len(data_file)

def calculate_two_paths(model, tokenizer, data_file, with_diagnosis):
    # print("\nNew Model\n")
    if with_diagnosis:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES and the DIAGNOSES AND FINDINGS of a patient.
You must review the knowledge base paths in CANDIDATE_PATHS based on concepts from PROGRESS_NOTES and select ONE PATH which can lead to one of the provided DIAGNOSES and FINDINGS.
Each path in CANDIDATE_PATHS follow the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
        '''
    else:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES of a patient.
You must review the knowledge base paths in CANDIDATE_PATHS based on concepts from PROGRESS_NOTES and select ONE PATH which can lead to a correct diagnosis.
Each path in CANDIDATE_PATHS follow the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
'''


    PROMPT_CONS += '''\n
Your OUTPUT must be just the path which can lead to valid diagnosis according to you. NO EXPLAINATION NEEDED.
'''

    rouge = evaluate.load('rouge')
    answers = []
    num_correct_picks = 0

    accs = []
    prec_list= []
    recall_list = []

    for data in tqdm(data_file[:1000]):
        # print("\nnew data\n")
        prompt = PROMPT_CONS + f"\nPROGRESS_NOTES: {data['progress_note']}"
        all_answers = [data['label_path'] if 'label_path' in data.keys() else data['answer']]
        all_answers.append(data['wrong_options'])
        
        if with_diagnosis:
            # print('adding diagnosis', with_diagnosis)
            note = openTrainFile()
            labels_note = note[data['id']][1].split(';')
            prompt += f"\nDIAGNOSES AND FINDINGS: {','.join(labels_note)}"

        prompt += f"\nCANDIDATE_PATHS:\n"
        random.shuffle(all_answers)
        # print("===all answers===", all_answers)
        for ind in range(2):
            prompt += f"{ind + 1}. {all_answers[ind]}\n"
        
        prompt += '''\nBased on the PROGRESS_NOTES, which PATH among the CANDIDATE_PATHS can lead to valid diagnosis ? Pick only ONE PATH and output the path that you have picked inside the <Answer>..</Answer> tag.
Output:\n'''

        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=64)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print("=====Prompt=====")

            # print(prompt)
            pred = output.split("\nOutput:")[1]

            # print("===Options===")

            # print("\n".join(all_answers))

            # print("===Answer===")

            # print(pred)

            # print("====Actual Answer====")

            # print(data['answer'])

            extracted_answer = None
            if len(pred.split("<Answer>")) > 1:
                extracted_answer = pred.split("<Answer>")[1].split("</Answer>")[0]
            
            # print("=====Extracted Answer=====")
            # print(extracted_answer)

            answers.append({
                'actual': data['answer'],
                'prediction': pred if extracted_answer == None else extracted_answer
            })

            gold_answer = data['answer'].replace("|", " ")
            gold_answer = gold_answer.replace("->", " ")

            predicted = answers[-1]['prediction'].replace("<Answer>", " ")
            predicted = predicted.replace("</Answer>", " ")
            predicted = predicted.replace("|", " ")
            predicted = predicted.replace("->", " ")
            notes_cui, gold_cui = get_path_labels(predicted, gold_answer)

            # print("====predicted====")
            # print(predicted)
            # print(notes_cui)

            # print("====actual====")
            # print(gold_answer)
            # print(gold_cui)
            if len(gold_cui) == 0:
                continue

            if len(notes_cui) == 0:
                accs.append(0)
                prec_list.append(0)
                recall_list.append(0)
            else:
                prec = len(gold_cui.intersection(notes_cui)) / len(notes_cui) 
                rec = len(gold_cui.intersection(notes_cui)) / len(gold_cui) 

                if prec + rec == 0:
                    accs.append(0)
                    prec_list.append(0)
                    recall_list.append(0)
                else:
                    acc = 2*(prec*rec) / (prec+rec)

                    accs.append(acc)
                    prec_list.append(prec)
                    recall_list.append(rec)

    print("Mean np", np.mean(accs))
    print("Precision np", np.mean(prec_list))
    print("Recall np", np.mean(recall_list))
    return rouge.compute(predictions=list(map(lambda x: x['prediction'], answers)),
                        references=list(map(lambda x: x['actual'], answers)))
            
def calculate_next_hop(model, tokenizer, data_file, with_diagnosis):
    print("\nNew Model\n")
    if with_diagnosis:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES and the DIAGNOSES AND FINDINGS of a patient.
You must review the knowledge base paths in CANDIDATE_PATHS based on concepts from PROGRESS_NOTES and select ONE PATH which can lead to one of the provided DIAGNOSES and FINDINGS.
Each path in CANDIDATE_PATHS follow the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
'''
    else:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES of a patient.
You are given a partial path and you must predict the next immediate node or relation edge of the path to reach a valid diagnosis based on the PROGRESS_NOTES of the patient.
The provided path follows the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.

For eg. if you are given a path like:

"nodeA->related_to|nodeB"

Because "nodeB" is a Node, the next hop to predict would be the relation edge through which "nodeB" would be connected to the next node in the graph.
Hence, your answer could be something like "->relation_edge".

For eg. if you are given a path like:

"nodeA->related_to|nodeB->subsection_of"

Because "subsection_of" is a Relation edge, the next hop to predict would be the Node to which "nodeB" would be connected to via edge "subsection_of".
Hence, your answer could be something like "|nodeC".

Your OUTPUT should be of one line and no explaination is needed.
'''

    rouge = evaluate.load('rouge')
    answers = []

    accs = []
    prec_list= []
    recall_list = []

    for data in tqdm(data_file[:1000]):
        # note = openTrainFile()
        # labels_note = note[data['id']][1].split(';')
        all_paths = get_data_label(data_file, data['id'])
        # print("data keys", data.keys())
        prompt = PROMPT_CONS + f"\nPROGRESS_NOTES: {data['progress_note']}"
        prompt += f"\nGiven the PATH {data['prompt'].split("Given the PATH ")[1].split(",")[0]}, predict the next hop of the PATH to reach a valid diagnosis. Output the next-hop inside the <Answer>..</Answer> tag.\nOutput:"
        
        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=64)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = output.split("\nOutput:")[1]

            extracted_answer = None
            if len(pred.split("<Answer>")) > 1:
                extracted_answer = pred.split("<Answer>")[1].split("</Answer>")[0]


            answers.append({
                'actual': data['answer'],
                'prediction': pred if extracted_answer == None else extracted_answer
            })

            # print("====Prompt====")
            # # # print(f"\nGiven the PATH {data['prompt'].split("Given the PATH ")[1].split(",")[0]}, predict the next hop of the PATH.\nOutput:")
            # print(prompt)
            # print("====Predicted Answer====")
            # print(pred)
            # print("====Actual Answer====")
            # print(data['answer'])
            # print("===== label path =====")
            # print(data['label_path'])
            # print("===== ground truth diagnoses =====")
            # print(labels_note)
            # print("===== all paths =====")
            # print('\n'.join(all_paths))

            # notes_cui, gold_cui = get_path_labels(pred, data['answer'])

            # if len(notes_cui) == 0 or len(gold_cui) == 0:
            #     # num_invalid += 1
            #     continue
            
            # prec = len(gold_cui.intersection(notes_cui)) / len(notes_cui) 
            # rec = len(gold_cui.intersection(notes_cui)) / len(gold_cui) 

            # if prec + rec == 0:
            #     accs.append(0)
            #     prec_list.append(0)
            #     recall_list.append(0)
            # else:
            #     acc = 2*(prec*rec) / (prec+rec)

            #     accs.append(acc)
            #     prec_list.append(prec)
            #     recall_list.append(rec)

    # print("Mean np", np.mean(accs))
    # print("Precision np", np.mean(prec_list))
    # print("Recall np", np.mean(recall_list))
    return rouge.compute(predictions=list(map(lambda x: x['prediction'], answers)),
                        references=list(map(lambda x: x['actual'], answers)))

def calculate_partial_completion(model, tokenizer, data_file, with_diagnosis):
    # print("\nNew Model\n")
    PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES of a patient.
You are given a partial knowledge base path and you must complete the rest of the path which can help with reaching a valid diagnoses based on the PROGRESS_NOTES of the patient.
The provided path follows the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
'''
# For eg. if you are given a path like:

# "nodeA->related_to|nodeB"

# Because "nodeA->related_to|nodeB->subsection_of|nodeC" could be a path which is valid based on the PROGRESS_NOTES, your answer would be "->subsection_of|nodeC".

# '''
    rouge = evaluate.load('rouge')
    answers = []

    for data in tqdm(data_file[:1000]):
        # note = openTrainFile()
        # labels_note = note[data['id']][1].split(';')
        all_paths = get_data_label(data_file, data['id'])
        # print("\ndata\n")
        prompt = PROMPT_CONS + f"\nPROGRESS_NOTES: {data['progress_note']}"
        prompt += f"\nGiven the PATH '{data['label_path'].removesuffix(data['answer'])}', complete the rest of the PATH which can help reaching a valid diagnoses for the patient. Output the completion of partial path inside the <Answer>..</Answer> tags. Output only one completion.\nOutput:"
        
        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=50)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = output.split("\nOutput:")[1]

            extracted_answer = None
            if len(pred.split("<Answer>")) > 1:
                extracted_answer = pred.split("<Answer>")[1].split("</Answer>")[0]


            answers.append({
                'actual': data['answer'],
                'prediction': pred if extracted_answer == None else extracted_answer
            })

            # print("====Prompt====")
            # # print(f"\nGiven the PATH '{data['label_path'].removesuffix(data['answer'])}', complete the rest of the PATH.\nOutput:")
            # print(prompt)
            # print("====Predicted Answer====")
            # print(pred)
            # print("====Actual Answer====")
            # print(data['answer'])
            # print("====extracted====")
            # print(extracted_answer)
            # print("===== ground truth diagnoses =====")
            # print(labels_note)
            # print("===== all paths =====")
            # print('\n'.join(all_paths))

    return rouge.compute(predictions=list(map(lambda x: x['prediction'], answers)),
                        references=list(map(lambda x: x['actual'], answers)))
            

def calculate_binary(model, tokenizer, data_file, with_diagnosis):
    if with_diagnosis:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES and the DIAGNOSES AND FINDINGS of a patient.
        You must review PATH based on concepts from PROGRESS_NOTES and answer if by using this PATH we can reach a valid conclusion belonging to DIAGNOSES AND FINDINGS.
        The PATH follows the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
        The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
        '''
    else:
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES of a patient.
        You must review the provided PATH based on concepts from PROGRESS_NOTES and answer if by using this PATH we can reach a correct diagnosis.
        The PATH follows the format: 'Node1->Relation|Node2->..' where 'Node1' and 'Node2' signify the concept nodes in the knowledge graph and 'Relation' signifies the relation between 'Node1' and 'Node2'. 
        The '->' symbol signifies a single hop from Node1 to Node2 via Relation edge.
        '''

    PROMPT_CONS += '''
Your answer needs to be a single word, either "YES" or "NO" depending on if the provided path can reach valid diagnosis or not.

For eg. consider the example path below:

"Node1->connnected_with|Node2->leads_to:Node3"

If you think that the above example path leads to a correct diagnosis according to the provided patient records, then your output must be: "YES", otherwise "NO".
Your Answer CANNOT be anything other than "YES" or "NO".    
'''
    rouge = evaluate.load('rouge')
    answers = []
    num_correct_picks = 0

    for data in tqdm(data_file[:]):
        # print("data", data)
        prompt = PROMPT_CONS + f"\nPROGRESS_NOTES: {data['progress_note']}"
        ans = data['label_path'] if 'label_path' in data.keys() else data['answer']
        wrong_choice = random.choice(data['wrong_options']) if isinstance(data['wrong_options'], list) else data['wrong_options']

        if with_diagnosis:
            # print('adding diagnosis', with_diagnosis)
            note = openTrainFile()
            labels_note = note[data['id']][1].split(';')
            prompt += f"\nDIAGNOSES AND FINDINGS: {','.join(labels_note)}"

        prompt1 = prompt + f"\nPATH: {ans}\n"
        
        prompt1 += '''\nBased on the above information, can we reach a valid diagnosis using the provided PATH ?\n
        Your answer must be EITHER "YES" or "NO".\n
        \nOutput: '''
        
        prompt2 = prompt + f"\nPATH: {wrong_choice}\n"
        
        prompt2 += '''\nBased on the above information, can we reach a valid diagnosis using the provided PATH ?\n
        Your answer must be EITHER "YES" or "NO".\n
        \nOutput: '''
        
        with torch.no_grad():
            model_inputs = tokenizer(prompt1, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=1)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print("=====Prompt=====")

            # print(prompt1)
            pred = output.split("\nOutput:")[-1]
            # print("====Actual Answer====")

            # print(ans)
            
            # print("===Answer===")
            # # print(output)
            # print(pred)

            answers.append({
                'actual': data['answer'],
                'prediction': pred 
            })

            model_inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=1)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print("=====Prompt2=====")

            # print(prompt2)
            pred = output.split("\nOutput:")[-1]

            # print("====Actual Answer2====")

            # print(wrong_choice)

            # print("===Answer2===")
            # # print(output)
            # print(pred)

            # print("====Actual Answer2====")

            # print(data['answer'])
            
            answers.append({
                'actual': data['answer'],
                'prediction': pred
            })


def main():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    args = parse_args()
    model_names = args.model_names
    data_file = args.data_file
    eval_metric = args.eval_metric
    with_diagnosis =  bool(args.with_diagnosis)
    
    if with_diagnosis:
        print("diagnosis is", with_diagnosis)

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    target_csv = []

    with open(data_file, "rb") as f:
            data = pickle.load(f)
    
    for model_path in tqdm(model_names):
        print("For Model", model_path)
        # if os.path.exists(model) and "merged_models" not in model and "DogeRM" not in model:
        #     model = get_last_checkpoint(model)
        #     print("checkpoint", model)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config).to("cuda:0")
        
        if os.path.exists(model_path):
            model = PeftModel.from_pretrained(model, model_path)
        # model = model.merge_and_unload()
        model.eval()
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        if eval_metric == "d1":
            target_csv.append(calculate_rouge(model, tokenizer, data, with_diagnosis))
            print("Rouge Scores:", target_csv[-1])
        elif eval_metric == "d3":
            # print("Next Hop", get_next_hop_predictions(model, tokenizer, data))
            print("next hop", calculate_next_hop(model, tokenizer, data, with_diagnosis))
        elif eval_metric == "d4":
            print("partial path complete", calculate_partial_completion(model, tokenizer, data, with_diagnosis))
        elif eval_metric == "d2":
            print("Binary Scores:", calculate_two_paths(model, tokenizer, data, with_diagnosis))
        elif eval_metric == "mcq":
            print("multiple choice",get_multiple_choice_answer(model, tokenizer, data, with_diagnosis))
        elif eval_metric == "d5":
            print("multiple paths", calculate_multiple_paths(model, tokenizer, data, with_diagnosis))
        else:
            print("Binary Scores:", calculate_binary(model, tokenizer, data, with_diagnosis))

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
if __name__ == '__main__':
    main()
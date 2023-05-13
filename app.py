from flask import Flask
from flask import jsonify
from flask import request
import requests
import logging
import os
import compress_fasttext
from razdel import sentenize, tokenize
import numpy as np
import re
from string import punctuation
from time import sleep
import trafilatura

app = Flask(__name__)
TOKEN = os.getenv('TOKEN')
TOKEN_SPACE = os.getenv('TOKEN_SPACE')
API_URL = "https://codemurt-qa-model2.hf.space/api/predict"
BEARER_SPACE = "Bearer " + TOKEN_SPACE
headers = {"Authorization": BEARER_SPACE, "Content-Type": "application/json"}

model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('model/geowac_tokens_sg_300_5_2020-100K-20K-100.bin')

logging.basicConfig(level=logging.INFO)

def kl_preprocess(sent):
    sent = sent.lower()
    res = re.findall('[а-яёa-z0-9]+', sent)
    tmp_sent = ""
    for i in res:
        tmp_sent += i
        tmp_sent += ' '
    return tmp_sent.strip()

def kl_tokenize(sentence):
    tokens = [_.text for _ in list(tokenize(sentence))]
    res = [token for token in tokens if token not in punctuation]
    return res

def cosine(u, v):
    if np.isnan(u).any() or np.isnan(v).any(): 
        return 0.0
    res = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return res

def embed(text):
    res = []
    tokens = kl_tokenize(text)
    for token in tokens:
        res.append(model[token])

    return np.mean(res, axis=0)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/onlyhtml', methods=["POST"])
def echo_only_html():
    try:
        req = request.get_json()
        html_req = req['html']
        question = req['question']
        logging.info(f'onlytext: request: {question}')
    except:
        logging.info(f'Failed. Incorrect input')
        return "Incorrect input"

    txt = trafilatura.extract(html_req)
    txt = txt.replace('\n', ' ')
    
    lst = [_.text for _ in list(sentenize(txt))]
    new_lst = []
    for sent in lst:
        new_lst.append(kl_preprocess(sent))

    embedded_data = [(embed(new_lst[i]), i) for i in range(len(new_lst))]
    
    indexes = set()

    def add_idx_to_set(idx):
        idx = int(idx)
        for i in range(idx - 3, idx + 3):
            if 0 <= i < len(lst):
                indexes.add(i)
    
    def get_context(set_indexes):
        ctx = ""
        for el in set_indexes:
            ctx += lst[el]
            ctx += " "
        return ctx

    def send_request(context):
        output = True
        while output:
            res = query({
                "data": [question, context]
            })
            logging.info(f'onlytext: send request: {res}')
            output = 'error' in res.keys()
            if output:
                sleep(3)
        return res['data'][0]

    def get_result(text):
        query = embed(text)
        res_lst = []
        cosines = [(cosine(x[0], query), x[1]) for x in embedded_data]
        vals = sorted(cosines, key=lambda x: x[0], reverse=True)
        for cos, cos_idx in vals[:3]:
            add_idx_to_set(int(cos_idx))
            curr_ctx = get_context(indexes)
            indexes.clear()
            curr_res = send_request(curr_ctx)
            res_lst.append((curr_res, curr_ctx))
        return res_lst
        
    final_res_lst = get_result(kl_preprocess(question))

    def clean_sent(sent):
        for sent_idx in range(len(sent)):
            if sent[sent_idx] not in punctuation:
                sent = sent[sent_idx:]
                break
        for sent_idx in range(len(sent) - 1, 0, -1):
            if sent[sent_idx] not in punctuation:
                sent = sent[:sent_idx + 1]
                break
        return sent
    
    for ctx_ans in final_res_lst:
        ctx_lst = [_.text for _ in list(sentenize(ctx_ans[1]))]
        ctx_ans[0]['answer'] = clean_sent(ctx_ans[0]['answer'])
        ctx_ans[0]['answer'] = ctx_ans[0]['answer'].strip()
        for ctx_sent in ctx_lst:
            if ctx_ans[0]['answer'] in ctx_sent:
                ctx_sent = clean_sent(ctx_sent)
                ctx_ans[0]['context'] = ctx_sent.strip()
                break
                
    return_final_res_lst = [_[0] for _ in final_res_lst]
    return_final_res_lst.sort(reverse=True, key=lambda x: x['score'])
    logging.info(f'onlytext: return answer: {return_final_res_lst}')
    indexes.clear()
    return return_final_res_lst

@app.route('/', methods=["GET"])
def hello_world():
    return "Hello user!"
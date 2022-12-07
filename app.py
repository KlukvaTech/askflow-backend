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
#API_URL = "https://api-inference.huggingface.co/models/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
API_URL = "https://codemurt-qa-model2.hf.space/api/predict"
# BEARER = "Bearer " + TOKEN
# headers = {"Authorization": BEARER}
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
    tmp_sent = tmp_sent.strip()
    return tmp_sent

def kl_tokenize(sentence):
    tokens = [_.text for _ in list(tokenize(sentence))]
    res = [token for token in tokens if token not in punctuation]
    return res

def cosine(u, v):
    res = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return res

def embed(text):
    if text == "":
        return 0.0
    if len(text.split()) <= 2:
        return 0.0 
    res = []
    tokens = kl_tokenize(text)
    for token in tokens:
        res.append(model[token])

    return np.mean(res, axis=0)

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.route('/onlytext', methods=["POST"])
def echo_only_text():
    try:
        req = request.get_json()
        logging.info(f'onlytext: request: {req}')
        txt = req['text']
        question = req['question']
    except:
        logging.info(f'Failed. Incorrect input')
        return "Incorrect input"

    #query({"inputs": {"question": "Turn", "context": "Turn! Turn! Turn!"}})

    txt = txt.replace('\n', '. ')
    lst = [_.text for _ in list(sentenize(txt))]
    new_lst = []
    for sent in lst:
        new_lst.append(kl_preprocess(sent))
    new_lst = [x for x in new_lst if x]
    embedded_data = [(embed(new_lst[i]), i) for i in range(len(new_lst))]
    
    indexes = set()

    # def add_idx_to_set(idx):
    #     idx = int(idx)
    #     for i in range(idx - 1, idx + 2):
    #         if 0 <= i < len(lst):
    #             indexes.add(i)

    def get_result(text):
        query = embed(text)

        cosines = [(cosine(x[0], query), x[1]) for x in embedded_data]
        print("got cosines")

        vals = sorted(cosines, key=lambda x: x[0])
        idx_ans = int(vals[-1][1])
        #add_idx_to_set(idx_ans)
        indexes.add(idx_ans)
    
    get_result(kl_preprocess(question))

    def get_context(set_indexes):
        ctx = ""
        for el in set_indexes:
            ctx += new_lst[el]
            ctx += " "
        return ctx
    
    context = get_context(indexes)

    output = True
    while output:
        # res = query({
        #     "inputs": {
        #         "question": question,
        #         "context": context
        #     }
        # })
        res = query({
            "data": [question, context]
        })
        logging.info(f'onlytext: send request: {res}')
        output = 'error' in res.keys()
        if output:
            sleep(3)
    res['context'] = context.strip()
    #res['answer'] = res['answer'].strip()
    res['answer'] = res['data'][0].strip()
    logging.info(f'onlytext: return answer: {res}')
    indexes.clear()
    return res

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

    #query({"inputs": {"question": "Turn", "context": "Turn! Turn! Turn!"}})

    txt = trafilatura.extract(html_req)

    new_txt = ""
    for i in range(1, len(txt)):
        if txt[i] == '\n' and (txt[i - 1] != '.' or txt[i - 1] != '!' or txt[i - 1] != '?' or txt[i - 1] != ';'):
            new_txt += ". "
        elif txt[i] == '\n':
            new_txt += " "
        else:
            new_txt += txt[i]
    lst = [_.text for _ in list(sentenize(new_txt))]
    new_lst = []
    for sent in lst:
        new_lst.append(kl_preprocess(sent))
    #new_lst = [x for x in new_lst if x]
    embedded_data = [(embed(new_lst[i]), i) for i in range(len(new_lst))]
    
    indexes = set()

    # def add_idx_to_set(idx):
    #     idx = int(idx)
    #     for i in range(idx - 1, idx + 2):
    #         if 0 <= i < len(lst):
    #             indexes.add(i)

    def get_result(text):
        query = embed(text)

        cosines = [(cosine(x[0], query), x[1]) for x in embedded_data]
        print("got cosines")

        vals = sorted(cosines, key=lambda x: x[0])
        idx_ans = int(vals[-1][1])
        #add_idx_to_set(idx_ans)
        indexes.add(idx_ans)
    
    get_result(kl_preprocess(question))

    def get_context(set_indexes):
        ctx = ""
        for el in set_indexes:
            ctx += lst[el]
            ctx += " "
        return ctx
    
    context = get_context(indexes)

    output = True
    while output:
        # res = query({
        #     "inputs": {
        #         "question": question,
        #         "context": context
        #     }
        # })
        res = query({
            "data": [question, context]
        })
        logging.info(f'onlytext: send request: {res}')
        output = 'error' in res.keys()
        if output:
            sleep(3)
    res['context'] = context.strip()
    #res['answer'] = res['answer'].strip()
    res['answer'] = res['data'][0].strip()
    logging.info(f'onlytext: return answer: {res}')
    indexes.clear()
    return res

@app.route('/allcontext', methods=["POST"])
def all_context():
    try:
        req = request.get_json()
        logging.info(f'onlytext: request: {req}')
        txt = req['text']
        question = req['question']
    except:
        logging.info(f'Failed. Incorrect input')
        return "Incorrect input"
    txt = txt.replace('\n', '. ')
    lst = [_.text for _ in list(sentenize(txt))]
    new_lst = []
    for sent in lst:
        new_lst.append(kl_preprocess(sent))
    new_lst = [x for x in new_lst if x]
    context = " ".join(new_lst)
    output = True
    while output:
        res = query({
            "data": [question, context]
        })
        logging.info(f'onlytext: send request: {res}')
        output = 'error' in res.keys()
        if output:
            sleep(3)
    res['context'] = txt.strip()
    res['answer'] = res['data'][0].strip()
    logging.info(f'onlytext: return answer: {res}')

    return res

@app.route('/baobab', methods=["POST"])
def baobab_text():
    dt = request.get_json()
    d = {'LOL KEK' : dt}
    return jsonify(d)

@app.route('/testing', methods=["POST"])
def answer_extension():
    d = {'LOL KEK' : "CHEBUREK"}
    return jsonify(d)

@app.route('/', methods=["GET"])
def hello_world():
    return "Hello user!"
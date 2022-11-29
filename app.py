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

app = Flask(__name__)
TOKEN = os.getenv('TOKEN')
API_URL = "https://api-inference.huggingface.co/models/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
BEARER = "Bearer " + TOKEN
headers = {"Authorization": BEARER}

model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('model/geowac_tokens_sg_300_5_2020-100K-20K-100.bin')

logging.basicConfig(level=logging.INFO)

def kl_preprocess(sent):
    sent = sent.lower()
    res = re.findall('[а-яё]+', sent)
    tmp_sent = ""
    for i in res:
        tmp_sent += i
        tmp_sent += ' '
    tmp_sent = tmp_sent.rstrip()
    return tmp_sent

def kl_tokenize(sentence):
    tokens = [_.text for _ in list(tokenize(sentence))]
    res = [token for token in tokens if token not in punctuation]
    return res

def cosine(u, v):
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

@app.route('/onlytext', methods=["POST"])
def echo_only_text():
    query({"inputs": {"question": "Turn", "context": "Turn! Turn! Turn!"}})
    req = request.get_json()
    logging.info(f'onlytext: request: {req}')
    txt = req['text']
    question = req['question']

    txt = txt.replace('\n', '. ')
    lst = [_.text for _ in list(sentenize(txt))]
    new_lst = []
    for sent in lst:
        new_lst.append(kl_preprocess(sent))
    new_lst = [x for x in new_lst if x]
    embedded_data = [(embed(new_lst[i]), i) for i in range(len(new_lst))]
    
    indexes = set()

    def add_idx_to_set(idx):
        idx = int(idx)
        for i in range(idx - 1, idx + 2):
            if 0 <= i < len(lst):
                indexes.add(i)

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
    
    def get_context_for_response(set_indexes):
        ctx = ""
        for el in set_indexes:
            ctx += new_lst[el]
            ctx += " "
        return ctx

    context = get_context(indexes)

    output = True
    while output:
        res = query({
            "inputs": {
                "question": question,
                "context": context
            }
        })
        logging.info(f'onlytext: send request: {res}')
        output = 'error' in res.keys()
        if output:
            sleep(10)
    res['context'] = get_context_for_response(indexes)
    res['answer'] = res['answer'].rstrip()
    logging.info(f'onlytext: return answer: {res}')
    indexes.clear()
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
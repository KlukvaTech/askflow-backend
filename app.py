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

# @app.route('/onlytext', methods=["POST"])
# def echo_only_text():
#     try:
#         req = request.get_json()
#         logging.info(f'onlytext: request: {req}')
#         txt = req['text']
#         question = req['question']
#     except:
#         logging.info(f'Failed. Incorrect input')
#         return "Incorrect input"

#     #query({"inputs": {"question": "Turn", "context": "Turn! Turn! Turn!"}})

#     txt = txt.replace('\n', '. ')
#     lst = [_.text for _ in list(sentenize(txt))]
#     new_lst = []
#     for sent in lst:
#         new_lst.append(kl_preprocess(sent))
#     new_lst = [x for x in new_lst if x]
#     embedded_data = [(embed(new_lst[i]), i) for i in range(len(new_lst))]
    
#     indexes = set()

#     # def add_idx_to_set(idx):
#     #     idx = int(idx)
#     #     for i in range(idx - 1, idx + 2):
#     #         if 0 <= i < len(lst):
#     #             indexes.add(i)

#     def get_result(text):
#         query = embed(text)

#         cosines = [(cosine(x[0], query), x[1]) for x in embedded_data]
#         print("got cosines")

#         vals = sorted(cosines, key=lambda x: x[0])
#         idx_ans = int(vals[-1][1])
#         #add_idx_to_set(idx_ans)
#         indexes.add(idx_ans)
    
#     get_result(kl_preprocess(question))

#     def get_context(set_indexes):
#         ctx = ""
#         for el in set_indexes:
#             ctx += new_lst[el]
#             ctx += " "
#         return ctx
    
#     context = get_context(indexes)

#     output = True
#     while output:
#         # res = query({
#         #     "inputs": {
#         #         "question": question,
#         #         "context": context
#         #     }
#         # })
#         res = query({
#             "data": [question, context]
#         })
#         logging.info(f'onlytext: send request: {res}')
#         output = 'error' in res.keys()
#         if output:
#             sleep(3)
#     res['context'] = context.strip()
#     #res['answer'] = res['answer'].strip()
#     res['answer'] = res['data'][0].strip()
#     logging.info(f'onlytext: return answer: {res}')
#     indexes.clear()
#     return res

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
    #txt = txt.replace('\n', ' ')
    new_txt = txt[0]
    for i in range(1, len(txt)):
        if txt[i] == '\n' and (txt[i - 1] != '.' or txt[i - 1] != '!' or txt[i - 1] != '?'):
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

    def add_idx_to_set(idx):
        idx = int(idx)
        for i in range(idx - 2, idx + 3):
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
        main_res = None
        main_ctx = None
        cosines = [(cosine(x[0], query), x[1]) for x in embedded_data]
        mx_score = -1.0    
        vals = sorted(cosines, key=lambda x: x[0], reverse=True)
        for cos, cos_idx in vals[:5]:
            add_idx_to_set(int(cos_idx))
            curr_ctx = get_context(indexes)
            indexes.clear()
            curr_res = send_request(curr_ctx)
            if curr_res['score'] > mx_score:
                main_res = curr_res
                mx_score = curr_res['score']
                main_ctx = curr_ctx
        return main_res, main_ctx
        #indexes.add(idx_ans)
    
    main_res, main_ctx = get_result(kl_preprocess(question))

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
    
    ctx_lst = [_.text for _ in list(sentenize(main_ctx))]
    main_res['answer'] = clean_sent(main_res['answer'])
    for ctx_sent in ctx_lst:
        if main_res['answer'] in ctx_sent:
            ctx_sent = clean_sent(ctx_sent)
            main_res['context'] = ctx_sent.strip()
            break
            

    #res['answer'] = res['answer'].strip()
    
    logging.info(f'onlytext: return answer: {main_res}')
    indexes.clear()
    return main_res

# @app.route('/onlyhtml_bert', methods=["POST"])
# def echo_only_html_bert():
#     try:
#         req = request.get_json()
#         html_req = req['html']
#         question = req['question']
#         logging.info(f'onlytext: request: {question}')
#     except:
#         logging.info(f'Failed. Incorrect input')
#         return "Incorrect input"

#     #query({"inputs": {"question": "Turn", "context": "Turn! Turn! Turn!"}})

#     txt = trafilatura.extract(html_req)
#     txt = txt.replace("\n", " ")
#     # new_txt = txt[0]
#     # for i in range(1, len(txt)):
#     #     if txt[i] == '\n' and (txt[i - 1] != '.' or txt[i - 1] != '!' or txt[i - 1] != '?'):
#     #         new_txt += ". "
#     #     elif txt[i] == '\n':
#     #         new_txt += " "
#     #     else:
#     #         new_txt += txt[i]
#     lst = [_.text for _ in list(sentenize(txt))]
#     new_lst = []
#     for sent in lst:
#         new_lst.append(kl_preprocess(sent))
#     #new_lst = [x for x in new_lst if x]

#     def mean_pooling(model_output, attention_mask):
#         token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#     #Encode text
#     def encode(texts):
#         # Tokenize sentences
#         encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

#         # Compute token embeddings
#         with torch.no_grad():
#             model_output = model_rubert(**encoded_input, return_dict=True)

#         # Perform pooling
#         embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

#         # Normalize embeddings
#         embeddings = F.normalize(embeddings, p=2, dim=1)
        
#         return embeddings
    
#     preprocessed_question = kl_preprocess(question)
#     query_emb = encode(preprocessed_question)
#     doc_emb = encode(new_lst)

#     scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

#     doc_score_pairs = list(zip(new_lst, scores, range(len(new_lst))))
#     doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#     indexes = set()

#     def add_idx_to_set(idx):
#         idx = int(idx)
#         for i in range(idx - 2, idx + 3):
#             if 0 <= i < len(lst):
#                 indexes.add(i)

#     def get_context(set_indexes):
#         ctx = ""
#         for el in set_indexes:
#             ctx += lst[el]
#             ctx += " "
#         return ctx

#     def send_request(context):
#         output = True
#         while output:
#             res = query({
#                 "data": [question, context]
#             })
#             logging.info(f'onlytext: send request: {res}')
#             output = 'error' in res.keys()
#             if output:
#                 sleep(3)
#         return res

#     mx_score = -1.0
#     for doc, score, doc_idx in doc_score_pairs[:5]:
#         add_idx_to_set(doc_idx)
#         curr_ctx = get_context(indexes)
#         indexes.clear()
#         curr_res = send_request(curr_ctx)
#         if curr_res['score'] > mx_score:
#             main_res = curr_res
#             mx_score = curr_res['score']
#             main_ctx = curr_ctx
    
#     response_output = main_res['data'][0]
#     response_output['context'] = main_ctx.strip()
#     response_output['answer'] = response_output['answer'].strip()
#     logging.info(f'onlytext: return answer: {main_res}')
#     return response_output

# @app.route('/allcontext', methods=["POST"])
# def all_context():
#     try:
#         req = request.get_json()
#         logging.info(f'onlytext: request: {req}')
#         txt = req['text']
#         question = req['question']
#     except:
#         logging.info(f'Failed. Incorrect input')
#         return "Incorrect input"
#     txt = txt.replace('\n', '. ')
#     lst = [_.text for _ in list(sentenize(txt))]
#     new_lst = []
#     for sent in lst:
#         new_lst.append(kl_preprocess(sent))
#     new_lst = [x for x in new_lst if x]
#     context = " ".join(new_lst)
#     output = True
#     while output:
#         res = query({
#             "data": [question, context]
#         })
#         logging.info(f'onlytext: send request: {res}')
#         output = 'error' in res.keys()
#         if output:
#             sleep(3)
#     res['context'] = txt.strip()
#     res['answer'] = res['data'][0].strip()
#     logging.info(f'onlytext: return answer: {res}')

#     return res

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
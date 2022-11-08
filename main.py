from flask import Flask
from flask import jsonify
from bs4 import BeautifulSoup
from bs4.element import Comment
from flask import request
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

@app.route('/text', methods=["POST"])
def echo_text():
    req = request.get_json()
    txt = text_from_html(req['text'])
    question = req['question']
    output = True
    while output:
        res = query({
            "inputs": {
                "question": question,
                "context": txt
            },
        })
        output = 'error' in res.keys()
    
    return jsonify(res)

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
    return "Ilya lox"
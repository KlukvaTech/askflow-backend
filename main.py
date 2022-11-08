from flask import Flask
from flask import jsonify
from bs4 import BeautifulSoup
from bs4.element import Comment
from flask import request

app = Flask(__name__)

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
    d = {'LOL KEK' : text_from_html(req['text'])}
    return jsonify(d)

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
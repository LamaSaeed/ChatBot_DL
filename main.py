import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

model = keras.models.load_model('chat_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)
    

def chat(msg):
    inp = msg
    max_len = 20
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                            truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            answer = np.random.choice(i['responses'])
    
    return answer





from flask import Flask, render_template, url_for, request
app = Flask(__name__)

@app.route('/')
def start():
    return render_template("start.html")

@app.route('/register/')
def register():
    return render_template("register.html")

@app.route('/chat/')
def  home():
    return render_template("chat.html")

@app.route("/get")
def get_bot_reponse():
    userText = request.args.get('msg')
    return str(chat(userText))

if __name__ == "__main__":
    app.run(debug=True)
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

import os
_BASE = os.path.dirname(os.path.abspath(__file__))

intents = json.loads(open(os.path.join(_BASE, 'intents.json'), encoding='utf-8').read())
words = pickle.load(open(os.path.join(_BASE, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(_BASE, 'classes.pkl'), 'rb'))
model = load_model(os.path.join(_BASE, "chatbot_simplilearnmodel.h5"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure I understand."
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure I understand."

app = FastAPI(title="Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    probability: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    intents_list = predict_class(req.message)
    reply = get_response(intents_list)
    intent = intents_list[0]['intent'] if intents_list else "unknown"
    prob = intents_list[0]['probability'] if intents_list else "0"
    return ChatResponse(response=reply, intent=intent, probability=prob)
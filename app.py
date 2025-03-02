import streamlit as st
import joblib as jb
import tensorflow as tf
import numpy as np
from tensorflow.preprocessing.sequence import pad_sequences
import re

model=tf.keras.models.load_model('emotion_model.h5')
tokenizer=jb.load('tokenizer.jb')
label_encoder=jb.load('label_encoder.jb')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"\S+", '', text).strip()
    return text

def predict_emotion(text):
    processed_text=preprocess_text(text)
    sequence=tokenizer.texts_to_sequences([processed_text])
    padded_sequence=pad_sequences(sequence, maxlen=100, padding='post')
    prediction=model.predict(padded_sequence)
    predicted_label=label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")

model = load_model()

st.title("LSTMood")
st.write("Check the tone of your messages!")

labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

if prompt := st.text_input(label="Type your message...", max_chars=200, autocomplete=None):
    preds = model.predict(tf.constant([prompt]))
    label = tf.argmax(preds, axis=1).numpy()[0]
    reply = f"Your mood is {labels[label]}"
    
    if label == 2:
        st.success(reply)
    elif label == 1:
        st.info(reply)
    else:
        st.error(reply)
import pickle
import pandas as pd
import streamlit as st

st.title('Fake News Detector')
with open('model.pkl','rb') as f:
    load_data = pickle.load(f)
with open('vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)



Text = st.text_area('Enter the message')
if st.button("Predict"):
    if Text.strip()=="":
        st.warning("please enter something")        
    else:
        transformer_text = vectorizer.transform([Text])

        prediction = load_data.predict(transformer_text)
        st.write("Raw Prediction:", prediction[0])
        if prediction[0] == 1:
            st.error('Fake')
        else:
            st.success('Real')


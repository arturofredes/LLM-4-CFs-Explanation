from openai import OpenAI
import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from predicting_functions import *

client = OpenAI()




with open(""".\Text Embedding Classifier\models\\tpot_model.pkl""", 'rb') as file:
    clf = pickle.load(file)

with open(""".\Text Embedding Classifier\models\\scaler.pkl""", 'rb') as file:
    scaler = pickle.load(file)


model_embeddings = tf.keras.models.load_model(""".\Text Embedding Classifier\models\\DNN_GPT_embeddings.h5""")

def zero_few_page():
    models = ['gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo']
    st.title('Zero/Few Shot Classifier')

    with st.form("my_form"):
        classifier = st.selectbox("Choose prompting strategy", ('Zero-Shot Learning', 'Few-Shot Learning'))
        role = st.radio('Select the role of the model', ['general assistant','expert hotelier','expert linguist'])
        reasoning = st.checkbox('Chain of thought (display reasoning)')
        confidence = st.checkbox('Confidence Level')
        model = st.radio("Model", models)
        iterations =  st.slider('Select a number of iterations to check consistency', min_value=1, max_value=5, value=1)

        review = st.text_area("Write a review about a hotel")


        submitted = st.form_submit_button("Submit")
        if submitted:
            if classifier == 'Zero-Shot Learning':
                shots = 'zero'
            elif classifier == 'Few-Shot Learning':
                shots = 'few'
            predictions = prompting(review, iterations=iterations, shots=shots, explanation = reasoning, model = model, confidence=confidence, role=role)
            print(predictions)
            # HTML and CSS to create a green box for the markdown
            st.markdown("""
            <style>
            .green-box {
                background-color: #90ee90;  /* Light green background */
                border: 1px solid green;
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            .yellow-box {
                background-color: #ffff99;  /* Light yellow background */
                border: 1px solid #ffcc00;  /* Darker yellow border */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)


            
            st.markdown("""
            <style>
            .red-box {
                background-color: #ffcccc;  /* Light red background */
                border: 1px solid #ff3333;  /* Darker red border */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            for output in predictions:
                print(output)
                lower=extract_from_output(output, 'class').lower()
                if 'positive' in lower:
                    # Markdown text inside a green box
                    st.markdown('<div class="green-box">'+output+'</div>', unsafe_allow_html=True)
                elif 'neutral'  in lower:
                    # Markdown text inside a green box
                    st.markdown('<div class="yellow-box">'+output+'</div>', unsafe_allow_html=True)
                elif 'negative' in lower:
                    # Markdown text inside a green box
                    st.markdown('<div class="red-box">'+output+'</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div>'+output+'</div>', unsafe_allow_html=True)

def embeddings_page():
    models = ['DNN','SVC + GaussianNB']
    st.title('Embeddings Classifier')
    with st.form("my_form"):
        model = st.radio("Model", models)
        review = st.text_area("Write a review about a hotel")
        #reasoning = st.checkbox('Display reasoning')
        submitted = st.form_submit_button("Submit")
        if submitted:
            embeddings = np.array(get_embedding_gpt(review)).reshape(1,1536)
            emb_scaled = scaler.transform(embeddings)
            if model == 'SVC':
                pred = clf.predict(emb_scaled)
            else:
                predictions = model_embeddings.predict(emb_scaled)
                pred = np.argmax(predictions, axis=1)
            if pred == 0:
                predictions = 'negative'
            elif pred == 1:
                predictions = 'neutral'
            else:
                predictions = 'positive'
            #st.subheader(predictions)

            # HTML and CSS to create a green box for the markdown
            st.markdown("""
            <style>
            .green-box {
                background-color: #90ee90;  /* Light green background */
                border: 1px solid green;
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("""
            <style>
            .yellow-box {
                background-color: #ffff99;  /* Light yellow background */
                border: 1px solid #ffcc00;  /* Darker yellow border */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)


            
            st.markdown("""
            <style>
            .red-box {
                background-color: #ffcccc;  /* Light red background */
                border: 1px solid #ff3333;  /* Darker red border */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            lower=predictions.lower()
            if 'positive' in lower:
                # Markdown text inside a green box
                st.markdown('<div class="green-box">'+predictions+'</div>', unsafe_allow_html=True)
            elif 'neutral' in lower:
                # Markdown text inside a green box
                st.markdown('<div class="yellow-box">'+predictions+'</div>', unsafe_allow_html=True)
            elif 'negative' in lower:
                # Markdown text inside a green box
                st.markdown('<div class="red-box">'+predictions+'</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div>'+predictions+'</div>', unsafe_allow_html=True)

            with st.expander("Embeddings"):
                st.text_area("Embeddings returned by text-embedding-ada-002 (first 100 values). Size: "+str(embeddings.shape),str(embeddings[0][:100]))


pages={"Zero/Few Shot Classifier": zero_few_page,
        "Embeddings Classifier" : embeddings_page}
selected_page=st.selectbox('Menu',pages.keys())
pages[selected_page]()
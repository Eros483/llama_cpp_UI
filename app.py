import llama_cpp
import streamlit as st
import os
import requests

model_url="https://drive.google.com/file/d/10aA4NhXJtJBo7asI6-SAfVRMi9PNurSu/view?usp=sharing"
model_path_gguf="llama-3.1-8b-instruct-q4_k_m.gguf"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path_gguf):
        st.info(f"Downloading model from {model_url}")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(model_path_gguf, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded succesfully")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}")
            st.stop()
    
    model=llama_cpp.Llama(model_path=model_path_gguf, chat_format="llama-2")

model = load_model()

inputQuery=st.text_input("Please enter your query: ")
temp=0.7

if inputQuery:
    response=model.create_chat_completion(messages=[{"role": "user", "content": inputQuery}], temperature=temp, max_tokens=256)
    assistant_reply = response['choices'][0]['message']['content']
    st.write("Assistant Reply: ")
    st.write(assistant_reply)
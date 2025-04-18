import streamlit as st
import transformers
import torch

model_id="unsloth/Meta-Llama-3.1-8B-Instruct"

@st.cache_resource
def load_pipeline():
    pipeline=transformers.pipeline(
        "text-generation", 
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return pipeline

pipeline=load_pipeline()

st.title("llama-3.1-8b version assistant")

input_query=st.text_input("Please enter your query: ")

if input_query:
    messages=[
        {"role": "system", "content": "You are a helpful assistant, who answers any query I have as precisely and concisely as you can."},
        {"role": "user", "content": input_query},
    ]

    prompt=pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators=[
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs= pipeline(
        prompt,
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )

    assistant_reply=outputs[0]["generated_text"][len(prompt):]
    st.write("Assistant reply: ")
    st.write(assistant_reply)

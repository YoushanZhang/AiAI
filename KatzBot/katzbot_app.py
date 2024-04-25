import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import Literal, Union
import base64
import pandas as pd

# Custom message class for managing chat history
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# Function to load CSS for styling the chatbot interface
def load_css():
    with open(r"C:\Users\keert\Downloads\static\static\styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelForCausalLM.from_pretrained('C:/Users/keert/Downloads/checkpoint-70000')

# Function to generate chatbot response
def generate_response(model, tokenizer, prompt):
    bot_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt', max_length=100, truncation=True)
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=10,
        top_p=0.7,
        temperature=0.8
    )
    chatbot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chatbot_response

def handle_submit():
    user_input = st.session_state.user_input
    if user_input:  # Check if there's any input to process
        chatbot_response = generate_response(model, tokenizer, user_input)
        st.session_state.history.append(Message("human", user_input))
        st.session_state.history.append(Message("ai", chatbot_response))
        st.session_state.user_input = ""  # Reset the input box

# Initialize or load session state
if 'history' not in st.session_state:
    st.session_state.history = []



def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


bot_image_path = r'C:\Users\keert\Downloads\static\static\ai_icon.png'  # Update this path
human_image_path = r'C:\Users\keert\Downloads\static\static\user_icon.png'  # Update this path

bot_image_base64 = get_image_base64(bot_image_path)
human_image_base64 = get_image_base64(human_image_path)
load_css()  # Apply the custom CSS

#strtaing divs 


st.title("Katzbot")

# Use a container for the chat history and apply the chat-container class
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.history:
        if chat.origin == 'ai':
            image_base64 = bot_image_base64
        else:  # Assuming any non-'ai' origin is 'human'
            image_base64 = human_image_base64

        div = f"""
    <div class="chat-row 
        {'' if chat.origin == 'ai' else 'row-reverse'}">
        <img class="chat-icon" src="data:image/png;base64,{image_base64}"
             width=32 height=32>
        <div class="chat-bubble
        {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
            &#8203;{chat.message}
        </div>
    </div>
    """
        st.markdown(div, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container div

# Add a fixed input area
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([4.5, 1])
user_input = col1.text_input("", value="", placeholder="Ask your question here", label_visibility="collapsed", key="user_input", on_change=handle_submit)
if col2.button("Submit", on_click=handle_submit):
    pass  # Button functionality handled by the callback
st.markdown('</div>', unsafe_allow_html=True)  # Close fixed-input-container div


# Add some space at the end
for _ in range(3):
    st.markdown("")

existing_df = pd.DataFrame()

# Read existing chat history CSV file if it exists
try:
    existing_df = pd.read_csv('session_chat_history.csv')
except FileNotFoundError:
    pass

# Concatenate existing chat history with current session state chat history
new_df = pd.concat([existing_df, pd.DataFrame(st.session_state.history)])

# Write the combined chat history to the CSV file
new_df.to_csv('session_chat_history.csv', index=False)

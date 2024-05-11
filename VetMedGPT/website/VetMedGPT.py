import streamlit as st
import streamlit as st
import requests
import json
import aiohttp
import asyncio

st.set_page_config(page_title = "VetmedGPT", page_icon = ":tada", layout = 'wide')

# Function to send user input to the API endpoint and get a response
async def chat(text):
    async with aiohttp.ClientSession() as session:
        async with session.post(
                'http://74.68.156.74:8000/predict',
                json={"text": text}
        ) as response:
            if response.status == 200:
                data = await response.json()
                bot_response = data.get("prediction", "").replace("</s>", "")
                return bot_response
            else:
                return "Error: Unable to fetch output from the API endpoint URL"


css= """
<style>
h1 {
    font-size: 35px; /* Adjust the size as needed */
    line-height: 1.2; /* Adjust the line height as needed */
    text-align: center;
    padding: 10px; /* Adjust the padding as needed */
}

.sidebar .sidebar-content {
    width: 300px; /* Adjust the sidebar width as needed */
}

.sidebar .css-ew7g7l {
    font-size: 36px; /* Adjust the button font size as needed */
    padding: 10px 16px; /* Adjust the button padding as needed */
    margin: 5px 0; /* Adjust the button margin as needed */
}
body {
    font-size: 25px;
}
p {
    font-size: 25px;
}
</style>
"""
# Inject the custom CSS
st.markdown(css, unsafe_allow_html=True)
# Define custom CSS styles

# Define navigation bar with buttons
st.sidebar.title('Navigation')


st.sidebar.button("Home")
left_column, right_column = st.columns(2)
with left_column:
          st.title("VetMedGPT: Generative Pre-trained Transformer for Veterinary Medicine Healthcare")
          st.write("Generative Pre-trained Transformer for Veterinary Medicine Healthcare - Specialized AI tool for initial diagnosis and first aid in animal health, enhancing accessibility and care quality")
    
with right_column:
        st.image(r"C:\Users\thiru\Updated.jpeg")
    

if st.sidebar.button("About"):    
        st.header("About VetMedGPT")
        st.write("VetMedGPT is an innovative AI model tailored specifically for veterinary medicine healthcare. Developed to address the limitations in existing AI support for animal health, VetMedGPT utilizes a vast dataset of veterinary knowledge for training. With its focus on initial diagnosis and first aid for animals, VetMedGPT aims to enhance accessibility and quality of care in veterinary science. This specialized tool promises to bridge the gap in AI applications for animal health, offering valuable support to both pet owners and veterinary professionals")
                    

    


if st.sidebar.button("View on GitHub"):
    # Redirect to GitHub link
            st.markdown("[GitHub Link](https://github.com/YoushanZhang/AiAI/tree/main/VetMedGPT)")

# Chat interface
st.subheader('chat with VetMedGPT')
user_input = st.text_input("You:", "")
if st.button("Send"):
        # Trigger chat function asynchronously
        bot_response = asyncio.run(chat(user_input))
        # Display bot response
        st.text_area("VetMedGPT:", value=bot_response, height=200, max_chars=None, key=None)

if st.sidebar.button("Contact Us"):
    st.markdown("""
    Name: Pinxue Lin  
    Email: plin3@mail.yu.edu  

    Name: Sayed Raheel  
    Email: shussai1@mail.yu.edu  

    Name: Tirupathi Kadari  
    Email: tkadari@mail.yu.edu  

    Name: Varun Biyyala  
    Email: vbiyyala@mail.yu.edu  

    Name: Sakshi Bennur  
    Email: sbennur@mail.yu.edu  

    Name: Jainam Bhansal  
    Email: jbhansal@mail.yu.edu  
    """)



    






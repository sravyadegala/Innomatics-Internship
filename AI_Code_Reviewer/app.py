from openai import OpenAI
import streamlit as st

f = open(r"C:\Users\NAVYA\Downloads\MY_PYTHON_PRACTICE\Innomatics_internship\backend\AI_Code_Reviewer\key.txt")
openai_api_key = f.read()
client = OpenAI(api_key = openai_api_key)

#########################################
st.title("🗿 Welcome to Python CODE REVIEWER")
st.subheader("Review your code here.")

###########################################

prompt = st.text_area("Enter your code")

if st.button("Generate") == True:
        st.balloons()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": """You are a helpful code assistant.
            you will take a python code as an user input. Your job role is to explain the bugs and fix the bug.
            and generate the correct code in output."""},
            {"role": "user", "content": prompt}
            ],
            temperature = 0.2
        )
        st.write(response.choices[0].message.content)

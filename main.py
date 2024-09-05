import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import json
import streamlit as st

config_file_path = 'config.json'

# Open and read the JSON file
with open(config_file_path, 'r') as file:
    config_data = json.load(file)

GOOGLE_API_Key = config_data['GOOGLE_API_KEY']

llm = ChatGoogleGenerativeAI(model = 'gemini-pro', google_api_key = GOOGLE_API_Key, temperature = 0.7)

prompt_template = """
You are a helpful assistant for the {os} system. Provide instructions for {query} for {os}.

Instructions:
"""

# Create a prompt template
prompt = PromptTemplate(
    input_variables = ['query', 'os'],
    template=prompt_template
)

# Create a LangChain LLMChain instance with the prompt and language model
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Generate instructions using the LLMChain
def get_instructions(query, os):
    response = llm_chain.invoke({'query': query, 'os': os})
    return response['text']

# Streamlit app
def main():
    st.title("Help System")
    query = st.text_input('Enter your query : ')

    # Dropdown for selecting the operating system
    os_options = ["Windows", "macOS", "Linux", "Android", "iOS"]
    selected_os = st.selectbox("Select your operating system:", os_options)

    # Button to get instructions
    if st.button("Get Instructions"):
        if query and selected_os:
            instructions = get_instructions(query, selected_os)
            st.write(f"Instructions for '{query}' on {selected_os}:")
            st.write(instructions)
        else:
            st.write("Please enter a query and select an operating system.")

if __name__ == "__main__":
    main()
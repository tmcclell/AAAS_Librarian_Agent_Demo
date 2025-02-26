import streamlit as st
import requests
import base64
import time
from dotenv import load_dotenv
import os
import json
import pandas as pd
import PyPDF2

# Load environment variables
load_dotenv(override=True)

# Endpoints from environment variables
AGENT_ENDPOINT = 'http://127.0.0.1:8005'


################################################################################
#                                   MAIN
################################################################################

st.set_page_config(
    page_title="AI Agent Service Demo",
    layout="wide",  # wide mode
)
st.title("Azure AI Agent Service Demos")

st.subheader("Sustainable AI Agent")
st.text('Agent to help developers understand the impact of their code on the environment using Azure AI Search, Code Intepreter and Function Calling.')

# --- 1. Session State Initialization (for data upload) ---
if "srch_messages" not in st.session_state:
    st.session_state["srch_messages"] = []

if "srch_thread_id" not in st.session_state:
    st.session_state["srch_thread_id"] = None
    # POST to your Data Upload Endpoint
    response = requests.post(
        f"{AGENT_ENDPOINT}/create_thread"
    )
    response.raise_for_status()
    data = response.json()
    st.session_state["srch_thread_id"] = data


def upload_file_to_agent(uploaded_file, thread_id):
    # Read the file content
    file_content = uploaded_file.read()      
    
    # If the file is a CSV or JSON, you might want to parse it
    if uploaded_file.type == "text/csv":
        file_content = uploaded_file.getvalue().decode('utf-8')
    elif uploaded_file.type == "application/json":
        file_content = json.dumps(json.load(uploaded_file))
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        file_content = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            file_content += page.extract_text()

    # Ensure file_content is a string or bytes
    if isinstance(file_content, str):
        file_content = file_content.encode('utf-8')

    # Send the file content to the model
    payload = {
        "thread_id": thread_id,  # Ensure thread_id is included
        "file_name": uploaded_file.name,
        "file_data": base64.b64encode(file_content).decode('utf-8'),  # Ensure file_data is base64 encoded
    }

    try:
        response = requests.post(
            f"{AGENT_ENDPOINT}/upload_file",
            json=payload,
            stream=True
        )
        if response.status_code == 200:
            partial_response = ""
            for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
                if chunk:
                    partial_response += chunk
                    st.session_state["srch_messages"].append({"role": "assistant", "content": partial_response})
            st.success("File uploaded successfully!")
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

# File Upload
uploaded_file = st.file_uploader("Upload a file for the AI Agent", type=["txt", "csv", "json", "pdf"])
if uploaded_file is not None:
    upload_file_to_agent(uploaded_file, st.session_state["srch_thread_id"])

# Display Existing Messages
for msg in st.session_state["srch_messages"]:
    if msg["role"] == "data":
        with st.expander("Data from Agent"):
            st.json(json.loads((msg["content"])))
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

# Chat Input + Streaming Response
prompt = st.chat_input("Calculate the SCI for the uploaded document.")
if prompt:
    st.session_state["srch_messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    agent_data = None
    with st.chat_message("assistant"):
        placeholder2 = st.markdown(body="")
        partial_response = ""

        payload = {
            "thread_id": st.session_state["srch_thread_id"],
            "message": prompt,
        }

        try:
            response = requests.post(
                f"{AGENT_ENDPOINT}/run_agent",
                json=payload,
                stream=True
            )
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
                    if chunk:
                        chunk = chunk.replace(
                            "<code><pre>",
                            "<br><br>💻🔧⚙️⚡<b>Code Interpreter Session:</b>\n```",
                        )
                        chunk = chunk.replace("</pre></code>", "\n```\n")
                        partial_response += chunk 
                        placeholder2.markdown(partial_response, unsafe_allow_html=True)
                        
            else:
                partial_response = f"Error from API: {response.status_code} - {response.text}"
                placeholder2.markdown(partial_response)
        except Exception as e:
            partial_response = f"Request failed: {e}"
            placeholder2.markdown(partial_response)

        placeholder2.markdown(partial_response, unsafe_allow_html=True)

    st.session_state["srch_messages"].append({"role": "assistant", "content": partial_response})
    if agent_data:
        st.session_state['srch_messages'].append({'role': 'data', 'content': agent_data})


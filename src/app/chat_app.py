import streamlit as st
import requests
import base64
import time
from dotenv import load_dotenv
import os
import json
import pandas as pd

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

st.subheader("Librarian Agent")
st.text('Agent acts a librarian and helps you find books based on Goodreads book data located in both an Azure AI Search Index, and an Azure SQL Database, and generate visualizations over this data using Code Interpreter.')

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


# Display Existing Messages
for msg in st.session_state["srch_messages"]:
    if msg["role"] == "data":
        with st.expander("Data from Agent"):
            st.json(json.loads((msg["content"])))
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

# Chat Input + Streaming Response
prompt = st.chat_input("Chat with your Librarian Agent")
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
                        # if '## AGENTDATA:' in chunk:
                        #     parts = chunk.split('## AGENTDATA:')
                        #     chunk = parts[-1]
                        #     agent_data = chunk
                        #     # placeholder3 = st.json(json.loads(chunk))
                        # else:
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

        
        # response = requests.post(
        #     f"{AGENT_ENDPOINT}/retrieve_last_response",
        #     json={'thread_id': st.session_state["srch_thread_id"]},
        #     stream=False
        # )
        # text = response.json().replace('<code><pre>', '<br><br>💻🔧⚙️⚡<b>Code Interpreter Session:</b>\n```python').replace('</pre></code>', '\n```\n')
        # partial_response = text
        placeholder2.markdown(partial_response, unsafe_allow_html=True)
        

    st.session_state["srch_messages"].append({"role": "assistant", "content": partial_response})
    if agent_data:
        st.session_state['srch_messages'].append({'role': 'data', 'content': agent_data})
        # with st.expander('Data from Agent'):
        #     st.json((json.loads(agent_data)))


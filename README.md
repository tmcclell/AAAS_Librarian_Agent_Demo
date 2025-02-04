# Azure AI Agent Service Demo

## Librarian Agent

---

### Repo Contents:

[Agent Creation Notebook](./src/agent_creation/create_librarian_agent.ipynb)

This repo contains sample code for building an Agent in the Azure AI Agent Service (part of Azure AI Foundry). This agent has access to two custom functions that allow it to **query an Azure AI Search index** and **query an Azure SQL database**. Beyond this, the agent has the ability to use **Code Interpreter** which enables it to author and execute custom python to do things like generate charts and graphs.

Recommend to create a new `.env` file in the root directory using `sample.env` as a reference template.


[Agent Endpoint (FastAPI)](./src/api/multi_agent/multi_agent.py)

To invoke this agent at an API endpoint we have provided a FastAPI wrapper which encapsulates the agent and supports streaming responses. 

This approach is convenient as it allows for integration of the agent in other applications and workflows.

```python -m uvicorn multi_agent:app```

[Chat Application (Streamlit)](./src/app/chat_app.py)

As a simple testing harness, we have provided a basic Streamlit app with a chat interface that handles creating new threads, executing runs, and displaying streamed agent responses.

```python -m streamlit run chat_app.py```
from fastapi import FastAPI, Request, HTTPException  # Import necessary modules from FastAPI  
from fastapi.responses import StreamingResponse  # Import StreamingResponse for streaming responses  
from dotenv import load_dotenv  # Import load_dotenv to load environment variables from a .env file  
import os  # Import os for operating system related functions  
import base64  # Import base64 for encoding and decoding base64 data  
from pydantic import BaseModel  # Import BaseModel from pydantic for data validation  
import threading  # Import threading for creating and managing threads  
import queue  # Import queue for creating a queue to handle data between threads  
import tempfile  # Import tempfile for creating temporary files and directories  
  
app = FastAPI()  # Create a FastAPI app instance  
load_dotenv(override=True)  # Load environment variables from a .env file  
  
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import MessageTextContent
from azure.ai.projects.models import CodeInterpreterTool, MessageAttachment
from azure.ai.projects.models import FilePurpose
from pathlib import Path

import base64
import logging
import json

from azure.ai.projects.models import (
    AgentEventHandler,
    FunctionTool,
    MessageDeltaChunk,
    RequiredFunctionToolCall,
    RunStep,
    RunStepStatus,
    SubmitToolOutputsAction,
    ThreadMessage,
    ThreadRun,
    ToolOutput,
)

from dotenv import load_dotenv  # Import load_dotenv to load environment variables from a .env file  
import os  # Import os for operating system related functions  
import base64  # Import base64 for encoding and decoding base64 data  
from pydantic import BaseModel  # Import BaseModel from pydantic for data validation  
import threading  # Import threading for creating and managing threads  
import queue  # Import queue for creating a queue to handle data between threads  
import tempfile  # Import tempfile for creating temporary files and directories  
import json
from io import StringIO
import pandas as pd
import uuid
import pandas as pd
import pyodbc
import json
import datetime
from typing import Any, Callable, Set, Dict, List, Optional
from agent_functions import agent_functions
import time

app = FastAPI()  # Create a FastAPI application instance

load_dotenv(override=True)  # Load environment variables from a .env file  

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ['AZURE_AI_FOUNDRY_CONNECTION_STRING'],
)

agent = project_client.agents.get_agent(os.environ['AGENT_ID'])

functions = FunctionTool(agent_functions)

import json


def stringify_code(obj) -> str:
    """Convert a code-interpreter input (str, dict, list, etc.) to a human-readable string."""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return json.dumps(obj, indent=2, ensure_ascii=False)
    elif isinstance(obj, (list, tuple)):
        # Join each item on its own line
        return "\n".join(stringify_code(item) for item in obj)
    else:
        # Fallback
        return str(obj)

class FileUploadRequest(BaseModel):  
    thread_id: str  # Ensure thread_id field is included
    file_name: str  
    file_data: str  

class RunAgentRequest(BaseModel):  
    thread_id: str  
    message: str  

class RetrieveResponseRequest(BaseModel):
    thread_id: str

class ThreadHistoryRequest(BaseModel):  
    thread_id: str
  

@app.post("/create_thread")  
def create_thread():  
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ['AZURE_AI_FOUNDRY_CONNECTION_STRING'],
    )
    thread = project_client.agents.create_thread()  # Create a new thread using the client  
    return thread.id  # Return the thread ID  

  

@app.post("/run_agent")
async def run_agent(request: RunAgentRequest):  
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ['AZURE_AI_FOUNDRY_CONNECTION_STRING'],
    )
    thread_id = request.thread_id  
    user_message = request.message 
    print(thread_id)
    print(user_message)
   

    def generate_response():  
        q = queue.Queue()  # Create a queue to handle data between threads 
        return_elements = [] 

        class CustomAgentEventHandler(AgentEventHandler):

            def __init__(self, functions: FunctionTool, project_client: AIProjectClient, thread_id: str) -> None:
                super().__init__()
                self.functions = functions
                self.project_client = project_client
                self.code_interpreter_active = False
                self.thread_id = thread_id
                self.queue = q
                self.return_elements = return_elements

            def on_message_delta(self, delta: "MessageDeltaChunk") -> None:
                # print(f"Text delta received: {delta.text}")
                self.queue.put(delta.text)
                pass

            def on_thread_message(self, message: "ThreadMessage") -> None:
                # print(f"ThreadMessage created. ID: {message.id}, Status: {message.status}")
                pass

            def on_thread_run(self, run: "ThreadRun") -> None:
                # print(f"ThreadRun status: {run.status}")

                if run.status == "failed":
                    print(f"Run failed. Error: {run.last_error}")
                    pass

                if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    tool_outputs = []
                    for tool_call in tool_calls:
                        # if isinstance(tool_call):
                        try:
                            if tool_call.type == "function":
                                display_text = ''
                                arguments = json.loads(tool_call.function.arguments)
                                # self.queue.put(f"<b>Function Call:</b> {tool_call}\n\n<br>")
                                if tool_call.function.name == "retrieve_documents_from_user_index":
                                    self.queue.put('Retrieving Documents from AI Search\n<br>')
                                    display_text = f'Searching for: <i><span style="color: limegreen;">{arguments["search_text"]}</span></i>\n\n<br>'
                                    self.queue.put(display_text)
                                elif tool_call.function.name=='run_query_on_table':
                                    # self.queue.put('Retrieving Records from SQL')
                                    display_text = 'Running query Against SQL\n<br>'
                                    # self.queue.put(display_text)
                                    query = arguments['sql_query']
                                    self.queue.put(f"Running SQL Query: \n```sql\n{query}\n```\n")
                              
                                output = functions.execute(tool_call)
                               
                                tool_outputs.append(
                                    ToolOutput(
                                        tool_call_id=tool_call.id,
                                        output=output,
                                    )
                                )
                        except Exception as e:
                            print(f"Error executing tool_call {tool_call.id}: {e}")

                    # print(f"Tool outputs: {tool_outputs}")
                    if tool_outputs:
                       
                        with self.project_client.agents.submit_tool_outputs_to_stream(
                            thread_id=run.thread_id, run_id=run.id, tool_outputs=tool_outputs, event_handler=self    
                        ) as stream:
                            stream.until_done()
                else:
                    pass

            def on_run_step(self, step: "RunStep") -> None:
                print(f"RunStep type: {step.type}, Status: {step.status}")
                if step.status==RunStepStatus.COMPLETED:
                    if 'tool_calls' in step.step_details:
                        for tc in step.step_details.tool_calls:
                            if tc.type=='code_interpreter':
                                self.code_interpreter_active = False
                                self.queue.put('</pre></code>')

                                for output in tc.code_interpreter.outputs:
                                    if output.type=='image':
                                        file_id = output.image.file_id
                                        data = project_client.agents.get_file_content(file_id)
                                        chunks = []
                                        for chunk in data:
                                            if isinstance(chunk, (bytes, bytearray)):
                                                chunks.append(chunk)
                                        
                                        combined_bytes = b"".join(chunks)
                                        encoded_image = base64.b64encode(combined_bytes).decode('utf-8') 
                                        data_url = f'<img width="750px" src="data:image/png;base64,{encoded_image}"/><br/>'
                                        self.queue.put(data_url)
                                        self.queue.put('<br/><br/>\n\n')



            def on_run_step_delta(self, delta):
                
                tool_calls = delta.delta.step_details.tool_calls
                for tc in tool_calls:
                    if tc.type=='code_interpreter':
                        if not self.code_interpreter_active:
                            self.code_interpreter_active = True
                            # self.queue.put('\n<b>Code Interpreter Session:</b>')
                            self.queue.put('<code><pre>'.encode('unicode_escape'))
                        code =  tc.code_interpreter.input
                        if code is not None:
                            code_str = stringify_code(code)

                            # Send as bytes
                            self.queue.put(code_str)
            

                # return super().on_run_step_delta(delta)

            def on_error(self, data: str) -> None:
                print(f"An error occurred. Data: {data}")

            def on_done(self) -> None:
                print("Stream completed.")

            def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
                print(f"Unhandled Event Type: {event_type}, Data: {event_data}")

        # Function to run the SDK code  
        def run_agent_code():  
            # Send the user message  
            
            
            message = project_client.agents.create_message(
                thread_id=thread_id,  
                role="user",  
                content=user_message,  
            )
            
            handler = CustomAgentEventHandler(functions, project_client, thread_id)  

            # Use the stream SDK helper with the EventHandler  
            with project_client.agents.create_stream(
                thread_id=thread_id, assistant_id=agent.id, event_handler=handler
            ) as stream:
                stream.until_done()
                time.sleep(1)

                # Indicate completion  
                q.put('<<DONE>>')  

        # Start the SDK code in a separate thread  
        sdk_thread = threading.Thread(target=run_agent_code)  
        sdk_thread.start()  

        # Read items from the queue and yield them  
        while True:  
            item = q.get()  
            if item == '<<DONE>>':  
                # yield '## AGENTDATA: ' + json.dumps(return_elements)
                break  
            if item == None:
                continue
            yield item

        sdk_thread.join()  

    return StreamingResponse(generate_response(), media_type="text/plain")  

@app.post("/retrieve_last_response")  
async def retrieve_last_response(request: RetrieveResponseRequest):  
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ['AZURE_AI_FOUNDRY_CONNECTION_STRING'],
    )
    try:  
        thread_id = request.thread_id  
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.environ['AZURE_AI_FOUNDRY_CONNECTION_STRING'],
        )
        messages = project_client.agents.list_messages(thread_id=thread_id)
        runs = project_client.agents.list_runs(thread_id=thread_id)

        objects = []
        last_msg = None
        for message in messages.data:
            if message.role =='user':
                last_msg = message.created_at
                break
            objects.append(message)
            last_msg = message.created_at

        for run in runs.data:
            if run.created_at >= last_msg:
                objects.append(run)
            else:
                break

        updated_objects = []
        for obj in objects:
            if obj.object=='thread.run':
                run_steps = project_client.agents.list_run_steps(thread_id=thread_id, run_id=obj.id).data
                for step in run_steps:
                    if step.type!= 'message_creation':
                        updated_objects.append(step)
            else:
                updated_objects.append(obj)

        updated_objects.sort(key=lambda x: x.created_at)
        updated_objects

        return_str = ''
        for obj in updated_objects:
            if obj.object =='thread.run.step':
                if obj.step_details.type == 'tool_calls':
                    if obj.step_details.tool_calls[0].type=='code_interpreter':
                        return_str += '<code><pre>'
                        return_str += obj.step_details.tool_calls[0].code_interpreter['input']
                        return_str += '</pre></code>'
            if obj.object =='thread.message':
                for content in obj.content:
                    if content.type=='text':
                        return_str += content.text.value
                    if content.type=='image_file':
                        data = project_client.agents.get_file_content(content.image_file.file_id)
                        chunks = []
                        for chunk in data:
                            if isinstance(chunk, (bytes, bytearray)):
                                chunks.append(chunk)
                        
                        combined_bytes = b"".join(chunks)
                        encoded_image = base64.b64encode(combined_bytes).decode('utf-8') 
                        data_url = f'<img width="750px" src="data:image/png;base64,{encoded_image}"/><br/>'
                        return_str += data_url


        return return_str
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to retrieve response: {str(e)}")

@app.post("/upload_file")
async def upload_file(request: FileUploadRequest):
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=os.environ['AZURE_AI_FOUNDRY_CONNECTION_STRING'],
    )
    try:
        # Define the CustomAgentEventHandler class
        class CustomAgentEventHandler(AgentEventHandler):

            def __init__(self, functions: FunctionTool, project_client: AIProjectClient, thread_id: str) -> None:
                super().__init__()
                self.functions = functions
                self.project_client = project_client
                self.code_interpreter_active = False
                self.thread_id = thread_id
                self.queue = queue.Queue()
                self.return_elements = []

            def on_message_delta(self, delta: "MessageDeltaChunk") -> None:
                self.queue.put(delta.text)
                pass

            def on_thread_message(self, message: "ThreadMessage") -> None:
                pass

            def on_thread_run(self, run: "ThreadRun") -> None:
                if run.status == "failed":
                    print(f"Run failed. Error: {run.last_error}")
                    pass

                if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls

                    tool_outputs = []
                    for tool_call in tool_calls:
                        try:
                            if tool_call.type == "function":
                                display_text = ''
                                arguments = json.loads(tool_call.function.arguments)
                                if tool_call.function.name == "retrieve_documents_from_user_index":
                                    self.queue.put('Retrieving Documents from AI Search\n<br>')
                                    display_text = f'Searching for: <i><span style="color: limegreen;">{arguments["search_text"]}</span></i>\n\n<br>'
                                    self.queue.put(display_text)
                                elif tool_call.function.name == 'run_query_on_table':
                                    display_text = 'Running query Against SQL\n<br>'
                                    query = arguments['sql_query']
                                    self.queue.put(f"Running SQL Query: \n```sql\n{query}\n```\n")
                              
                                output = functions.execute(tool_call)
                                tool_outputs.append(
                                    ToolOutput(
                                        tool_call_id=tool_call.id,
                                        output=output,
                                    )
                                )
                        except Exception as e:
                            print(f"Error executing tool_call {tool_call.id}: {e}")

                    if tool_outputs:
                        with self.project_client.agents.submit_tool_outputs_to_stream(
                            thread_id=run.thread_id, run_id=run.id, tool_outputs=tool_outputs, event_handler=self    
                        ) as stream:
                            stream.until_done()
                else:
                    pass

            def on_run_step(self, step: "RunStep") -> None:
                print(f"RunStep type: {step.type}, Status: {step.status}")
                if step.status == RunStepStatus.COMPLETED:
                    if 'tool_calls' in step.step_details:
                        for tc in step.step_details.tool_calls:
                            if tc.type == 'code_interpreter':
                                self.code_interpreter_active = False
                                self.queue.put('</pre></code>')

                                for output in tc.code_interpreter.outputs:
                                    if output.type == 'image':
                                        file_id = output.image.file_id
                                        data = project_client.agents.get_file_content(file_id)
                                        chunks = []
                                        for chunk in data:
                                            if isinstance(chunk, (bytes, bytearray)):
                                                chunks.append(chunk)
                                        
                                        combined_bytes = b"".join(chunks)
                                        encoded_image = base64.b64encode(combined_bytes).decode('utf-8') 
                                        data_url = f'<img width="750px" src="data:image/png;base64,{encoded_image}"/><br/>'
                                        self.queue.put(data_url)
                                        self.queue.put('<br/><br/>\n\n')

            def on_run_step_delta(self, delta):
                tool_calls = delta.delta.step_details.tool_calls
                for tc in tool_calls:
                    if tc.type == 'code_interpreter':
                        if not self.code_interpreter_active:
                            self.code_interpreter_active = True
                            self.queue.put('<code><pre>'.encode('unicode_escape'))
                        code = tc.code_interpreter.input
                        if code is not None:
                            code_str = stringify_code(code)
                            self.queue.put(code_str)

            def on_error(self, data: str) -> None:
                print(f"An error occurred. Data: {data}")

            def on_done(self) -> None:
                print("Stream completed.")

            def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
                print(f"Unhandled Event Type: {event_type}, Data: {event_data}")

        # Send the JSON payload to the agent using the same thread
        payload = {
            "thread_id": request.thread_id,
            "file_name": request.file_name,
            "file_data": request.file_data,
        }
        message = project_client.agents.create_message(
            thread_id=request.thread_id,
            role="user",  # Use 'user' as the role
            content=json.dumps(payload),
        )

        # Use the same thread to send the payload to the model
        handler = CustomAgentEventHandler(functions, project_client, request.thread_id)
        with project_client.agents.create_stream(
            thread_id=request.thread_id, assistant_id=agent.id, event_handler=handler
        ) as stream:
            stream.until_done()

        return {"message": "File content sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to send file content: {str(e)}")



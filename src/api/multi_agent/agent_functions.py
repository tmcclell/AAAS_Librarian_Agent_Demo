import json
import datetime
from typing import Any, Callable, Set, Dict, List, Optional

from openai import AzureOpenAI
import openai
import os
import time
from dotenv import load_dotenv
import pandas as pd
import pyodbc
import json
import datetime

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchField,  
    VectorSearch,  
    HnswAlgorithmConfiguration, 
    VectorSearchProfile,
    
)

from azure.search.documents.models import VectorizedQuery

load_dotenv(override=True)

def generate_embeddings(text, model_name=None, key_based_auth=True):
    """
    Generates embeddings for the given text using the specified embeddings model provided by OpenAI.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        embeddings (list): The embeddings generated for the given text.
    """

    # Configure OpenAI with Azure settings
    openai.api_type = "azure"
    openai.api_base = os.environ['AOAI_ENDPOINT']
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.environ['AOAI_KEY']

    client = AzureOpenAI(
        azure_endpoint=os.environ['AOAI_ENDPOINT'], api_key=os.environ['AOAI_KEY'], api_version="2023-03-15-preview"
    )

    embedding_model = os.environ['AOAI_EMBEDDINGS_MODEL']
    if model_name is not None:
        embedding_model = model_name

    # Initialize variable to track if the embeddings have been processed
    processed = False
    # Attempt to generate embeddings, retrying on failure
    while not processed:
        try:
            # Make API call to OpenAI to generate embeddings
            response = client.embeddings.create(input=text, model=embedding_model)
            processed = True
        except Exception as e:  # Catch any exceptions and retry after a delay
            # logging.error(e)
            print(e)

            # Added to handle exception where passed context exceeds embedding model's context window
            if 'maximum context length' in str(e):
                text = text[:int(len(text)*0.95)]

            time.sleep(5)

    # Extract embeddings from the response
    embeddings = response.data[0].embedding
    return embeddings

def retrieve_documents_from_user_index(search_text:str, record_count=10, filter_statement='') -> List[Dict[str, str]]:
    """
    Retrieves Goodreads book description records from an Azure AI Search index using vector search. \
    This function performs a vector-based search on an Azure Cognitive Search index to \
    retrieve and return relevant records based on the provided search text. \ 
    Optionally, a filter statement can be provided to further restrict the search space, and you can specify a number of documents to be returned.

    :param search_text: The input search query text to generate embeddings and perform the search.
    :type search_text: str
    :param record_count: The maximum number of records to return. Defaults to 10.
    :type record_count: int, optional
    :param filter_statement: Optional filter statement that can be used to further restrict search space authored using OData syntax.
    :type filter_statement: str, optional
    :return: A list of dictionaries representing the search results. Each dictionary contains details about individual books.
    :rtype: list
    :raises KeyError: If the required environment variables `SEARCH_KEY` or `SEARCH_ENDPOINT` are not set.

    """


    # Get the search key, endpoint, and service name from environment variables
    search_key = os.environ['SEARCH_KEY']
    search_endpoint = os.environ['SEARCH_ENDPOINT']

    # Create a SearchIndexClient object
    credential = AzureKeyCredential(search_key)
    client = SearchClient(endpoint=search_endpoint, index_name=os.environ['SEARCH_INDEX_NAME'], credential=credential)

    embedding = generate_embeddings(search_text)

    # To learn more about how vector ranking works, please visit https://learn.microsoft.com/azure/search/vector-search-ranking
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=record_count, fields="embeddings")

    try:

        if len(filter_statement)>0:
            results = client.search(  
                vector_queries= [vector_query],
                search_text=search_text,
                top=record_count,
                select=['Description', 'Name', 'PublishYear', 'Authors', 'ISBN', 'Language', 'Publisher'],
                filter=filter_statement
            )  
            results = list(results)
            return json.dumps(results)
        else:
        
            results = client.search(  
                vector_queries= [vector_query],
                search_text=search_text,
                top=record_count,
                select=['Description', 'Name', 'PublishYear', 'Authors', 'ISBN', 'Language', 'Publisher']
            )  
            results = list(results)
            return json.dumps(results)
    except Exception as e:
        return str(e)

#def run_query_on_table(sql_query):  
    """  
    Runs a SQL query against a the Goodreads SQL data (book ratings).  
      
    :param sql_query: The SQL query to be executed.  
    :type sql_query: str  
    :return: The results of the query as a pandas DataFrame.  
    :rtype: pandas.DataFrame  
    :raises Exception: If any step fails, the exception is raised.  
    """  
      
    try:  
        # Database connection details from environment variables  
        server = os.environ["SQL_SERVER"]  
        database = os.environ["SQL_DATABASE"]  
        username = os.environ["SQL_USERNAME"]  
        password = os.environ["SQL_PASSWORD"]  
  
        conn_str = (  
            "DRIVER={ODBC Driver 18 for SQL Server};"  
            f"SERVER={server}.database.windows.net,1433;"  
            f"DATABASE={database};UID={username};PWD={password};"  
            "Encrypt=yes;TrustServerCertificate=no;"  
        )  
        conn = pyodbc.connect(conn_str)  
  
        # Format the query to ensure it runs against the specified table  
        print(sql_query)
        
  
        # Execute the query  
        df = pd.read_sql(sql_query, conn)  
  
        # Close the connection  
        conn.close()  
  
        return json.dumps(df.to_dict(orient='records'))
  
    except Exception as e:  
        print(f"An error occurred: {e}")  
        return str(e)
        raise  

# Statically defined user functions for fast reference
agent_functions: Set[Callable[..., Any]] = {
    #run_query_on_table,
    retrieve_documents_from_user_index
}
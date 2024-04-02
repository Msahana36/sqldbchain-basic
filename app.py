from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain.memory import ConversationBufferMemory
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import constants


os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

# Connection with db
os.environ["SQL_SERVER_USERNAME"] = constants.SQL_SERVER_USERNAME
os.environ["SQL_SERVER_ENDPOINT"] = constants.SQL_SERVER_ENDPOINT
os.environ["SQL_SERVER_PASSWORD"] = constants.SQL_SERVER_PASSWORD  
os.environ["SQL_SERVER_DATABASE"] = constants.SQL_SERVER_DATABASE

#Setup with Langsmith
os.environ["LANGCHAIN_TRACING_V2"] = constants.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = constants.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = constants.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = constants.LANGCHAIN_PROJECT


st.set_page_config(
    page_title="MagicSQL",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

llm = ChatOpenAI(temperature=0, model="gpt-4", streaming=True)

# db driver selection and setup
driver = '{ODBC Driver 18 for SQL Server}'
odbc_str = 'mssql+pyodbc:///?odbc_connect=' \
                'Driver='+driver+ \
                ';Server=tcp:' + os.environ["SQL_SERVER_ENDPOINT"]+'.database.windows.net;PORT=1433' + \
                ';DATABASE=' + os.environ["SQL_SERVER_DATABASE"] + \
                ';Uid=' + os.environ["SQL_SERVER_USERNAME"]+ \
                ';Pwd=' + os.environ["SQL_SERVER_PASSWORD"] + \
                ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;'

db_engine = create_engine(odbc_str)
db = SQLDatabase(db_engine)
memory = ConversationBufferMemory(llm=llm,memory_key='history', return_messages=True, output_key='result')
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, top_k=10, memory=memory)

prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """You are an MS SQL expert. Given an input question, first create a syntactically correct MS SQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 10 results using the TOP clause as per MS SQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in square brackets ([]) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CAST(GETDATE() as date) function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:

CREATE TABLE [Logs] (
	[GUID] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
	[TimeStamp] DATETIMEOFFSET NULL, 
	[SourceSystem] NVARCHAR(100) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
	[SourceApplication] NVARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
	[SourceModule] NVARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
	[Type] NVARCHAR(50) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
	[Tags] NVARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL, 
	[Description] NVARCHAR(max) COLLATE SQL_Latin1_General_CP1_CI_AS NULL
)
--------
 
 """),
                    MessagesPlaceholder("history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )

avatar_image = "Magicappicon.png"
user_image = "usericon.png"



starter_message = "How can I help you?"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]
    
    
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar=avatar_image).write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user", avatar=user_image).write(msg.content)
    memory.chat_memory.add_message(msg)

st.cache_resource(ttl="2h")   
def chatbot():
        
    if prompt := st.chat_input(placeholder=starter_message):
        st.chat_message("user", avatar=user_image).write(prompt)
        with st.chat_message("assistant", avatar=avatar_image):
            with st.spinner("Thinking..."):
                response = db_chain.invoke(
                {"query": prompt, "history": st.session_state.messages},
                include_run_info=True,
            )
            st.session_state.messages.append(AIMessage(content=response["result"]))
            st.write(response["result"])  
            memory.save_context({"query": prompt}, {"result":response["result"]})
            st.session_state["messages"] = memory.buffer
            run_id = response["__run"].run_id
        
chatbot()
        
        
        
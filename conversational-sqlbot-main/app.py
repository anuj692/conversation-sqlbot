import streamlit as st
from pathlib import Path #get absolute path
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pandas as pd
import re
from sqlalchemy import text

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt=["Use SQLLite 3 Database- Student.db","Connect to you MySQL Database"]

selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MySQL Host")
    mysql_user=st.sidebar.text_input("MYSQL User")
    mysql_password=st.sidebar.text_input("MYSQL password",type="password")
    mysql_db=st.sidebar.text_input("MySQL database")
else:
    db_uri=LOCALDB

api_key=st.sidebar.text_input(label="GRoq API Key",type="password", value=os.getenv("GROQ_API_KEY", ""))

if not db_uri:
    st.info("Please enter the database information and uri")
    st.stop()

if not api_key:
    st.info("Please add the groq api key")
    st.stop()

## LLM model
llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.3-70b-versatile",streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))   
    
if db_uri==MYSQL:
    db=configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db=configure_db(db_uri)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Upload CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file to add to the database", type=["csv"])

if uploaded_file is not None:
    table_name = st.sidebar.text_input("Table Name", value=uploaded_file.name.split('.')[0])
    if st.sidebar.button("Load Data to DB"):
        with st.spinner("Loading data..."):
            try:
                df = pd.read_csv(uploaded_file)
                valid_table_name = table_name.replace(" ", "_").replace("-", "_")
                
                # We need a writeable engine to upload (the langchain one is read-only for SQLite)
                if db_uri == LOCALDB:
                    dbfilepath = (Path(__file__).parent/"student.db").absolute()
                    write_engine = create_engine(f"sqlite:///{dbfilepath}")
                else:
                    write_engine = create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
                    
                df.to_sql(valid_table_name, con=write_engine, if_exists="replace", index=False)
                st.sidebar.success(f"Data loaded successfully into table '{valid_table_name}'!")
                
                # Clear cache so the agent sees the new table
                st.cache_resource.clear()
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")

## toolkit
toolkit=SQLDatabaseToolkit(db=db,llm=llm)

custom_prefix = """
You are an agent designed to interact with a SQL database.
You can execute SELECT queries to help the user.
If the user asks you to modify data (INSERT, UPDATE, DELETE, CREATE, DROP), DO NOT use the sql_db_query tool to execute it!
Instead, you must output the exact SQL query required in a markdown block like this:
```sql
<YOUR_SQL_QUERY_HERE>
```
and ask the user for approval.
"""

agent=create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="zero-shot-react-description",
    prefix=custom_prefix
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if "pending_query" in st.session_state and st.session_state["pending_query"]:
    st.info("The agent proposed the following database modification. Please approve or reject it.")
    st.code(st.session_state["pending_query"], language="sql")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve & Execute"):
        with st.spinner("Executing..."):
            try:
                if db_uri == LOCALDB:
                    dbfilepath = (Path(__file__).parent/"student.db").absolute()
                    write_engine = create_engine(f"sqlite:///{dbfilepath}")
                else:
                    write_engine = create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
                
                with write_engine.begin() as conn:
                    conn.execute(text(st.session_state["pending_query"]))
                
                st.success("Query executed successfully!")
                st.session_state.messages.append({"role": "assistant", "content": f"Executed query successfully:\n```sql\n{st.session_state['pending_query']}\n```"})
                st.session_state["pending_query"] = None
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Execution failed: {e}")
                
    if col2.button("❌ Reject"):
        st.session_state["pending_query"] = None
        st.session_state.messages.append({"role": "assistant", "content": "The proposed query was rejected."})
        st.rerun()

user_query=st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
        
        # Check if response contains DML
        sql_match = re.search(r"```(?:sql)?\n?(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER).*?```", response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            # Extract the content inside the markdown block
            sql_query = sql_match.group(0)
            sql_query = re.sub(r"^```(?:sql)?\n?", "", sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r"\n?```$", "", sql_query).strip()
            st.session_state["pending_query"] = sql_query
            st.rerun()

 # Custom footer styling (Fixed at the very bottom)
custom_footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 64%;
        transform: translateX(-50%);
        width: 100%;
        background: linear-gradient(135deg, #1f1f1f, #292929);
        text-align: center;
        padding: 12px;
        font-size: 14px;
        color: white;
        border-top: 2px solid #ffffff33;
        font-family: Arial, sans-serif;
        z-index: 1000;
    }
    </style>
    <div class="footer">
        Developed by <b style="color: #f8c700;">Anuj Bhardwaj</b>
    </div>
"""

# Ensure footer appears at the very bottom of the page
st.markdown(custom_footer, unsafe_allow_html=True)       




import streamlit as st
from langchain.llms import OpenAIChat
from langchain.llms import OpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.document_loaders import ConcurrentLoader

from langchain.prompts import PromptTemplate
from pathlib import Path
import faiss
from langchain.vectorstores import FAISS
import pickle
from pathlib import Path
import json
import datetime
from datetime import datetime
import os

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
import base64
import PyPDF2
from io import BytesIO
from typing import Any, Dict, List
from PyPDF2 import PdfReader
import re
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# datetime object containing current date and time
now = datetime.now()


# Set the title of the Streamlit app
st.title("`LangChain`")

name_project = st.sidebar.text_input(
    "Project Name",
)
# Using object notation

add_radio = st.sidebar.text_input(
            "Key"
)
os.environ["OPENAI_API_KEY"] = add_radio

uploaded_files = st.file_uploader(label = "Add text file !",accept_multiple_files=True)


import streamlit as st


if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""

if "input_git_repo" not in st.session_state:
    st.session_state["input_git_repo"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "memory" not in st.session_state:

    st.session_state['memory']=ConversationBufferMemory(memory_key="chat_history", input_key='human_input')
if "chat_history" not in st.session_state:

    st.session_state['chat_history']=""


input_text_git_repo = st.text_input("You: ", st.session_state["input_git_repo"], key="input_git_repo",
                            placeholder="Input git repo ...",
                            )

import os
from glob import glob
import pandas as pd
#only works with python
def get_function_name(code):
    """
    Extract function name from a line beginning with "def "
    """
    assert code.startswith("def ")
    return code[len("def "): code.index("(")]

def get_until_no_space(all_lines, i) -> str:
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, i + 10000):
        if j < len(all_lines):
            if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
                ret.append(all_lines[j])
            else:
                break
    return "\n".join(ret)

def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    whole_code = open(filepath).read().replace("\r", "\n")
    all_lines = whole_code.split("\n")
    for i, l in enumerate(all_lines):
        if l.startswith("def "):
            code = get_until_no_space(all_lines, i)
            function_name = get_function_name(code)
            yield {"code": code, "function_name": function_name, "filepath": filepath}




def model(docs,metadatas):


    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    print("inside model")
    template = """You are a chatbot having a conversation with a human.

    These are code files of the company, you might have to go through them and answer with code snippets and with analysis.
    Given the following extracted parts of a long document and a question, create a final answer. Always remember to return snippets of code.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"],
        template=template
    )


    embeddings = OpenAIEmbeddings()

    file_name = str(docs[0])[10:30]+str(str(docs[-1])[10:30])+".pkl"
    file_name = re.sub(r'[^\w]', '', file_name)
    file_name=file_name.replace('?', "")
    if os.path.isfile(file_name) :
#             st.success("loading vectorstore")
        index = faiss.read_index("fin_docs.index")

        with open(file_name, "rb") as f:
            store = pickle.load(f)

        store.index = index


    else:
        st.success("creating vectorstore")
        store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
        faiss.write_index(store.index, "fin_docs.index")
        store.index = None
        with open(file_name, "wb") as f:
            pickle.dump(store, f)
        index = faiss.read_index("fin_docs.index")

        with open(file_name, "rb") as f:
            store = pickle.load(f)

        store.index = index



    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # Modify model_name if you have access to GPT-4
    memory = ConversationBufferMemory(memory_key="chat_history", input_key='human_input')
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", memory=st.session_state.memory, prompt=prompt)


    return chain,store,prompt, memory


@st.cache_data()
def read_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    import os
    for page in pdf.pages:
        text = page.extract_text()

        # Merge hyphenated words
#         text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
#         text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
#         text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output

@st.cache_data()
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)

    return text

@st.cache_data()
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")

    return text


from git.repo.base import Repo

def save_files(git_link):
    import shutil
    # Directory name
    global bbb
    bbb = os.path.join(os.getcwd(), "Content/")

    from git import rmtree
    #change the path
    if os.path.exists("Content"):
        rmtree("Content")

#     dir_path =
#     dir_path.rmdir()
  

    os.mkdir("Content")
    # note: for this code to work, the openai-python repo must be downloaded and placed in your root directory
    Repo.clone_from(git_link,bbb)


def read_files(git_link):
    print(git_link)
    data = []
    sources = []
    # get user root directory


#     # path to code repository directory

    code_files = DirectoryLoader(bbb, glob="**/*.py", loader_cls=PythonLoader)
    documents = code_files.load()
    print(len(documents))

    all_funcs = []
    for code_file in code_files:
        funcs = list(get_functions(code_file))
        for func in funcs:
            all_funcs.append(func)

    print("Total number of functions extracted:", len(all_funcs))
    metadata = []
    docs=[]
    for i in range(len(all_funcs)):
        metadata.extend([{"source": "file name: "+all_funcs[i]['filepath']+" function name: "+all_funcs[i]['function_name']}])
        docs.extend([all_funcs[i]['code']])
    return docs,metadata

if st.button('Submit'):
    save_files(input_text_git_repo)
if input_text_git_repo:
    docs,metadatas = read_files(input_text_git_repo)
    chain,store,PROMPT,memory = model(docs,metadatas)
    print(chain)
    print(store)



# if uploaded_files:
#     docs,metadatas = read_files(uploaded_files)
# #
# #     st.info(metadatas)
#     chain,store,PROMPT,memory = model(docs,metadatas)
#     print("done")

def get_text():


    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your Coding AI assistant here! Ask me anything ...",
                            )

    return input_text

user_input = get_text()
print("asking input")
print(st.session_state["input"])
if user_input:

#     st.info("=========================="+output+"========================================")
    do = store.as_retriever().get_relevant_documents(user_input)
    output = chain({"input_documents": do, "human_input": user_input})
#
    st.success("Answer: "+output['output_text']+"\nSource: " + str(dict(output['input_documents'][0])['metadata']))

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output['output_text']+"\nSource: " +str(dict(output['input_documents'][0])['metadata']))
#     st.session_state['memory'] += memory.buffer
#     st.session_state['chat_history'] +=memory.buffer

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info('USER: '+ st.session_state["past"][i])
        st.success('AI: '+ st.session_state["generated"][i])
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)



add_radio = st.sidebar.subheader(
        "About"
    )
add_radio = st.sidebar.markdown(
        "https://langchain.readthedocs.io/en/latest/index.html"
    )


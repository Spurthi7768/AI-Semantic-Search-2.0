import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pinecone
import joblib
from langchain.llms import OpenAI
import numpy as np

st.title("📄 AI Semantic Search 2.0 💻")

if 'secret' not in st.session_state:
    st.session_state.secret=st.secrets['apikey']

user_secret=st.text_input(label=":black[OpenAI API Key]",placeholder="Paste your OpenAI API Key" ,type='password')
pinecone_secret=st.text_input(label=":black[PineCone API Key]",placeholder="Paste your PineCone API Key" ,type='password')
pinecone_env_secret=st.text_input(label=":black[PineCone Env]",placeholder="Paste your PineCone Environment Value")
if user_secret:
    openai.api_key=user_secret
    st.session_state.secret=user_secret
if pinecone_secret:
    if pinecone_env_secret:
        pinecone.init(api_key=pinecone_secret,environment=pinecone_env_secret)

@st.cache_data
def load_data():
    reader = PdfReader('data/living planet report.pdf')
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

data=load_data()


def load_pinecone_data(data):
    if pinecone_secret:
    # Initialize Pinecone connection
        pinecone_index_name = 'semantic-search' 
        # connect to index
        pinecone_index = pinecone.Index(pinecone_index_name)

        if pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(pinecone_index_name, dimension=1536)            
            count = 0  #  use the count to create unique IDs
            batch_size = 32  # process everything in batches of 32
            for i in range(0, len(data), batch_size):
                # set end position of batch
                i_end = min(i+batch_size, len(data))
                # get batch of lines and IDs
                lines_batch = data[i: i+batch_size]
                ids_batch = [str(n) for n in range(i, i_end)]
                # create embeddings
                res = openai.Embedding.create(input=lines_batch, engine="text-embedding-ada-002")
                embeds = [record['embedding'] for record in res['data']]
                # prep metadata and upsert batch
                meta = [{'text': line} for line in lines_batch]
                to_upsert = zip(ids_batch, embeds, meta)
                # upsert to Pinecone
                pinecone_index.upsert(vectors=list(to_upsert)) 

        return pinecone_index
    

pinecone_index=load_pinecone_data(data)
svm_model = joblib.load('model/svm_model.pkl')        


def search_article(search_query):
    xq = openai.Embedding.create(input=search_query, engine="text-embedding-ada-002")['data'][0]['embedding']
   
    predicted_index = svm_model.predict(np.array(xq).reshape(1, -1))[0]
    retrieved_document = pinecone_index.fetch(ids=[str(predicted_index)])
    retrieved_document = pinecone_index.fetch(ids=[str(predicted_index)])
    ans=str(retrieved_document['vectors'][str(predicted_index)]['metadata']['text'])
    return ans




search_query=st.text_input(label=":blue[Search your Query]",placeholder="Please, search the article with...")

search_button=st.button(label="Run",type="primary")

if search_query:
    if search_button:
        answer=search_article(search_query)

        st.write(answer)
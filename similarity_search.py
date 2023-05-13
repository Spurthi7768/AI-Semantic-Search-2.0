import numpy as np
from sklearn import svm
import joblib
import time
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

openai.api_key=""  #Add OpenAI API Key Here

#Load the pdf file
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

#Create embeddings
embeddings = []

for paragraph in texts:
    embedding=openai.Embedding.create(input=paragraph, engine="text-embedding-ada-002")['data'][0]['embedding']    
    embeddings.append(embedding)
    time.sleep(2)

embeddings_list=embeddings

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings_list)

# Initialize and train SVM model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(embeddings_np, range(len(embeddings_np)))

# Save the SVM model to disk
svm_model_path = "model/svm_model.pkl"
joblib.dump(svm_model, svm_model_path)

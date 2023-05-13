# AI-Semantic-Search-2.0
A webapp which can perform AI Semantic Search on the document provided by the user

## Prerequisite
Please keep the API Key for your OpenAI account handy. Instructions can be found [here](https://platform.openai.com/account/api-keys)
Please keep the API Key for your Pinecone account along with the environment name. Instructions can be found [here](https://docs.pinecone.io/docs/quickstart)

## Instructions to use
* Create a virtual environment in python
* Activate the virtual environment
* Install the packages from requirements.txt file
* Run the command : streamlit run app.py

## Data 
The data used is LIVING PLANET REPORT 2022 : https://wwfint.awsassets.panda.org/downloads/embargo_13_10_2022_lpr_2022_full_report_single_page_1.pdf

## Libraries
* Frontend: Streamlit
* Backend: LangChain, OpenAI and Pinecone
 
 The custom similiarity search has been implemented used for prediction of the most similar ID and the vector associated with the particular ID is fetched 
 from the Pinecone vector database. The algorithm used is Support Vector Machine which performs better than the K-Nearest Neighbor according to the article
 [here](https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb)

 
 ## Demo Link
 https://youtu.be/8L9frU4n8TU


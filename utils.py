import os
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.vectorstores import Pinecone
# from langchain.vectorstores import FAISS
# import faiss
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from datasets import Dataset
import pandas as pd
from athina.loaders import Loader
from athina.evals import DoesResponseAnswerQuery
from athina.keys import AthinaApiKey, OpenAiApiKey

# --- Initial Setup ---
def setup_environment():
    OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
    AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))
    os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY') #for local use
    embeddings = OpenAIEmbeddings()
    return embeddings

# --- Indexing ---
def load_and_split_documents(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    return documents

# --- Pinecone Vector Database ---
def create_pinecone_index(embeddings,documents):
  pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"),)
  index_name = 'my-index'
  if index_name not in pc.list_indexes().names:
      pc.create_index(
          name=index_name,
          dimension=1536,
          metric="cosine",
          spec=ServerlessSpec(cloud="aws", region="us-east-1"),
      )
  vectorstore = Pinecone.from_documents(
          documents=documents, embedding=embeddings, index_name=index_name
      )
  return vectorstore

# --- FAISS Vector Database (Optional) ---
# def create_faiss_index(embeddings, documents):
#     vectorstore = FAISS.from_documents(documents, embeddings)
#     return vectorstore

# --- Retriever ---
def create_retriever(vectorstore):
    retriever = vectorstore.as_retriever()
    return retriever

# --- RAG Chain ---
def create_rag_chain(retriever):
    llm = ChatOpenAI()
    template = """
    You are a helpful assistant that answers questions based on the provided context.
    Use the provided context to answer the question.
    Question: {input}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever,  "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Evaluation ---
def prepare_evaluation_data(rag_chain, question, retriever):
    response = []
    contexts = []
    for query in question:
      response.append(rag_chain.invoke(query))
      contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
    data = {
        "query": question,
        "response": response,
        "context": contexts,
    }
    return data
def dataset_from_dict(data):
  dataset = Dataset.from_dict(data)
  return dataset
def create_dataframe(dataset):
    df = pd.DataFrame(dataset)
    return df
def convert_context(df):
  df_dict = df.to_dict(orient='records')
  for record in df_dict:
    if not isinstance(record.get('context'), list):
        if record.get('context') is None:
          record['context'] = []
        else:
          record['context'] = [record['context']]
  return df_dict

def load_data_from_dict(df_dict):
  dataset = Loader().load_dict(df_dict)
  return dataset
def run_eval(dataset):
  eval_results = DoesResponseAnswerQuery(model="gpt-4o").run_batch(data=dataset).to_df()
  return eval_results
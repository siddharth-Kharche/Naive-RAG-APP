import streamlit as st
import os
from utils import (
    setup_environment,
    load_and_split_documents,
    create_pinecone_index,
    create_retriever,
    create_rag_chain,
    prepare_evaluation_data,
    dataset_from_dict,
    create_dataframe,
    convert_context,
    load_data_from_dict,
    run_eval,
)
# --- Streamlit App ---
st.title("Naive RAG App")

# --- Initialize ---
with st.sidebar:
  st.header("Configuration")
  if 'embeddings' not in st.session_state:
      with st.spinner("Loading Embeddings..."):
        st.session_state['embeddings'] = setup_environment()

  uploaded_file = st.file_uploader("Upload your CSV Context", type=['csv'])
  if uploaded_file:
    file_path = 'context.csv' #define the path
    with open(file_path,"wb") as f:
      f.write(uploaded_file.getvalue()) #writes the uploded file
    if 'documents' not in st.session_state:
        with st.spinner("Loading Documents"):
          st.session_state['documents'] = load_and_split_documents(file_path)
    if 'vectorstore' not in st.session_state:
      with st.spinner("Creating Vectorstore"):
          st.session_state['vectorstore'] = create_pinecone_index(st.session_state['embeddings'],st.session_state['documents'])
    if 'retriever' not in st.session_state:
      st.session_state['retriever'] = create_retriever(st.session_state['vectorstore'])
    if 'rag_chain' not in st.session_state:
      st.session_state['rag_chain'] = create_rag_chain(st.session_state['retriever'])

# --- User Input & Response ---
question = st.text_input("Enter your question:")
if question:
    if 'rag_chain' in st.session_state:
        with st.spinner("Generating Response..."):
            response = st.session_state['rag_chain'].invoke(question)
        st.subheader("Response:")
        st.write(response)

# --- Evaluation ---
if st.sidebar.button("Evaluate"):
    if 'rag_chain' in st.session_state and 'retriever' in st.session_state:
        with st.spinner("Preparing Data for Evaluation..."):
           data = prepare_evaluation_data(st.session_state['rag_chain'], [question],st.session_state['retriever'] )
        with st.spinner("Creating Dataset..."):
          dataset = dataset_from_dict(data)
        with st.spinner("Creating Dataframe..."):
           df = create_dataframe(dataset)
        with st.spinner("Converting Context..."):
           df_dict = convert_context(df)
        with st.spinner("Loading Data from Dictionary..."):
            eval_dataset = load_data_from_dict(df_dict)
        with st.spinner("Running Evaluation..."):
            eval_results = run_eval(eval_dataset)
        st.subheader("Evaluation Results:")
        st.dataframe(eval_results)
    else:
        st.warning("Please upload data and generate a response before evaluation.")
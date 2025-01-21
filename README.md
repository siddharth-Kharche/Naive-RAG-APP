Here's a README file for your project:

---

# Naive RAG App

This is a simple app that utilizes a **Retrieval-Augmented Generation (RAG)** approach to answer questions based on the context provided in a CSV file. The app uses **Pinecone** for vector storage, **LangChain** for document processing and retrieval, and **OpenAI** embeddings for contextual understanding.

## File Structure

```
streamlit_app/
├── app.py          # Main Streamlit application
├── utils.py        # Helper functions (e.g., vectorstore creation)
├── context.csv     # Your data file
├── .streamlit/     # Config folder for Streamlit
│   └── secrets.toml # Store API keys
├── requirements.txt # Project dependencies
```

## Setup Instructions

### 1. Install Dependencies

To run the app, first install the necessary libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 2. API Keys

You'll need to securely store your API keys in the `.streamlit/secrets.toml` file. Create the file and add the following:

```toml
OPENAI_API_KEY = "your_openai_key"
ATHINA_API_KEY = "your_athina_key"
PINECONE_API_KEY = "your_pinecone_key"
```

These keys will be used to authenticate the API calls to OpenAI, Athina, and Pinecone.

### 3. Run the Application

Once the dependencies are installed and the API keys are set, you can run the app:

```bash
streamlit run app.py
```

### 4. App Usage

- **Upload a CSV File**: The app accepts a CSV file, which will be used as the source of context for answering questions. 
- **Ask Questions**: Enter your question in the provided input box to get a response based on the uploaded context.
- **Evaluate**: Use the "Evaluate" button to run an evaluation on the question-answer pairs and view the results in a DataFrame.

## File Details

### `app.py`

This is the main Streamlit application that handles the UI and user interactions. It includes:

- File upload functionality for CSV context.
- Text input for asking questions.
- Evaluation trigger to assess the model's performance.

### `utils.py`

Contains helper functions for:

- Setting up the environment.
- Loading and splitting documents.
- Creating Pinecone vector store.
- Creating the retriever and RAG chain.
- Preparing and running the evaluation.

### `context.csv`

The CSV file that contains your context data. The app expects this file to have columns of text that will be used as the context for answering queries.

### `.streamlit/secrets.toml`

A configuration file that securely stores your API keys. It ensures that sensitive data (API keys) are not exposed in the code.

### `requirements.txt`

Lists all the required Python libraries for the app to run, including Streamlit, LangChain, Pinecone, etc.

## How It Works

1. **Environment Setup**: The `setup_environment()` function loads the necessary API keys and sets up the OpenAI embeddings for document processing.
2. **Document Loading**: The `load_and_split_documents()` function loads the CSV file and splits the text into smaller chunks for embedding.
3. **Vector Database**: The app supports Pinecone for vector storage. If you prefer to use FAISS, you can uncomment the relevant code in `utils.py`.
4. **Retriever and RAG Chain**: The `create_retriever()` and `create_rag_chain()` functions create the retrieval system and the RAG pipeline, respectively, to answer user queries.
5. **Evaluation**: The evaluation process takes the question, context, and response and runs an evaluation to assess the quality of the answer.

## Important Notes

- **API Keys**: Never commit your API keys to version control. Use `.streamlit/secrets.toml` to securely store them.
- **FAISS**: The FAISS implementation is optional and commented out. To use FAISS, uncomment the relevant lines in `utils.py` and ensure you have `faiss-gpu` installed.
- **Data Handling**: The app expects the uploaded CSV file to have columns that are interpretable as context. Ensure your data is in the correct format.

## Future Improvements

- Add more RAG techniques and customizable evaluation metrics.
- Improve error handling and logging.
- Extend the UI to allow for more interactive features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This should provide a good starting point for your project documentation! Let me know if you'd like to add anything else!

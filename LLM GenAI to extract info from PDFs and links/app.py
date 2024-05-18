import os
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#---------------- Functions -----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def create_documents_from_chunks(chunks, source):
    documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]
    return documents

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
#------------------------------------------------------------------------#


# Load environment variables
load_dotenv()

# Streamlit UI setup
st.title("Local ChatGPT")
st.write("You can upload your PDFs, enter your URLs, and ask your questions 😊")

st.sidebar.title("Your Article URLs")

# Input for the number of URLs
num_urls = st.sidebar.number_input("Enter number of URLs", min_value=1, max_value=10, value=3)

# Input URLs
urls = []
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

# Input PDFs
with st.sidebar:
    st.subheader("Your PDF Files")
    pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

# Process button
process_url_clicked = st.sidebar.button("Process")
index_file_path = "faiss_index"

# Placeholder for main content
main_placeholder = st.empty()

# Initialize LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load urls data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    url_raw_text = loader.load()

    # Load and process PDFs
    raw_pdf_text = get_pdf_text(pdf_docs)
    pdf_chunks = get_text_chunks(raw_pdf_text)
    pdf_raw_text = create_documents_from_chunks(pdf_chunks, source="PDF")

    # Combine text chunks from URLs and PDFs
    data = url_raw_text + pdf_raw_text

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)


    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)


    # Save the FAISS index
    vectorstore_openai.save_local(index_file_path)

# Query input
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_file_path):
        # Initialize embeddings again for loading the FAISS index
        embeddings = OpenAIEmbeddings()

        # Load the FAISS index with dangerous deserialization allowed
        vectorstore = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)

        # Create the RetrievalQAWithSourcesChain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Get the result
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)

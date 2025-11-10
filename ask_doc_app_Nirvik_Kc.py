import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa import RetrievalQA

# Function to generate response
def generate_response(uploaded_file, openai_api_key, query_text):
    if uploaded_file is not None:
        # Read and split file content
        documents = [uploaded_file.read().decode()]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create vectorstore
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        # Create QA chain
        llm = OpenAI(openai_api_key=openai_api_key)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        return qa.run(query_text)


# Streamlit UI
st.set_page_config(page_title="Ask the Doc App")
st.title("📄 Ask the Doc App")

uploaded_file = st.file_uploader("Upload a text file", type="txt")
query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary.",
    disabled=not uploaded_file
)

result = []
with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key", type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit", disabled=not (uploaded_file and query_text)
    )
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner("Calculating..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(result[-1])

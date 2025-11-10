import streamlit as st
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ----------------------------
# Function to generate response
# ----------------------------
def generate_response(uploaded_file, openai_api_key, query_text):
    if uploaded_file is not None:
        # Read file content
        document_text = uploaded_file.read().decode("utf-8", errors="ignore")

        # Split document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(document_text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create vectorstore
        db = Chroma.from_texts(texts, embeddings)

        # Create retriever
        retriever = db.as_retriever()

        # Create QA chain
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Run the query
        response = qa.run(query_text)
        return response

# Streamlit UI
st.set_page_config(page_title="Ask the Doc App", page_icon="📄")
st.title("📄 Ask the Doc App")

st.markdown("Upload a text file and ask any question about its contents!")

uploaded_file = st.file_uploader("Upload a text file", type="txt")

query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary or question about the document.",
    disabled=not uploaded_file,
)

# Collect user input in a form
result = []
with st.form("qa_form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        disabled=not (uploaded_file and query_text),
    )
    submitted = st.form_submit_button(
        "Submit", disabled=not (uploaded_file and query_text)
    )

    if submitted:
        if not openai_api_key.startswith("sk-"):
            st.error("Please enter a valid OpenAI API key (starts with sk-).")
        else:
            with st.spinner("Analyzing document..."):
                try:
                    response = generate_response(uploaded_file, openai_api_key, query_text)
                    result.append(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Display response
if len(result):
    st.success("✅ Response:")
    st.info(result[-1])

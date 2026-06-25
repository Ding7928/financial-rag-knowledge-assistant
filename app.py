import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="Financial RAG Knowledge Assistant")

st.title("Financial RAG Knowledge Assistant")
st.write("Upload a financial document and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document..."):
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Generating answer..."):
            retrieved_docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            prompt = ChatPromptTemplate.from_template(
                """
                You are a financial document assistant.
                Answer the user's question using only the context below.
                If the answer is not in the context, say you do not know.

                Context:
                {context}

                Question:
                {question}
                """
            )

            messages = prompt.format_messages(
                context=context,
                question=question
            )

            response = llm.invoke(messages)

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Retrieved Sources")
        for i, doc in enumerate(retrieved_docs, 1):
            page_number = doc.metadata.get("page", "N/A")
            st.markdown(f"**Source {i}: Page {page_number}**")
            st.write(doc.page_content[:800])
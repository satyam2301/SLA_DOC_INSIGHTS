import json
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Chatbot RAG chain handling multiple questions and storing conversation history
def generate_chat_answer(doc_text, question):
    # Step 1: Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(doc_text)

    # Step 2: Generate embeddings for each chunk using Google Gemini model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    # Step 3: Set up the retriever for searching relevant document chunks
    retriever = vectorstore.as_retriever()

    # Step 4: Define the prompt template for regular chatbot question-answering
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use ten sentences maximum and keep the answer more concise and to the point ans mostly one liner.
    Question: {question}
    Context: {context}
    Answer:
    """

    # Step 5: Create the prompt using the template
    prompt = ChatPromptTemplate.from_template(template)

    # Step 6: Set up the generative model (Google Gemini 1.5 Pro)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
    output_parser = StrOutputParser()

    # Step 7: Set up the RAG chain combining retrieval and response generation
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    # Step 8: Invoke the RAG chain with the given question
    result = rag_chain.invoke(question)

    return result
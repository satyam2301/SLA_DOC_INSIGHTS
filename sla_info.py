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

from pdf_extractor import extract_text_and_tables_from_pdf

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def generate_sla_info(doc_text):
    # Step 2: Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(doc_text)

    # Step 3: Generate embeddings for each chunk using Google Gemini model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    # Step 4: Set up the retriever for searching relevant document chunks
    retriever = vectorstore.as_retriever()

    # Step 5: Define the prompt template for SLA-related question-answering with specific instructions for detailed answers
    chat_template = """
    You are an assistant focused on providing detailed answers to specific questions based on the context provided.
    Answer the following questions based on the context. Provide detailed answers for each question. If the answer is not available, say "I don't know."

    1. What is the SLA? or What is the name of SLA?
    2. What are the names of the parties involved?
    3. What system is concerned in this SLA?
    4. Explain the description of the SLA in detail?
    5. What metrics are associated with this SLA?
    6. What is the exact page number of the SLA?

    Context: {context}

    Please provide each answer in a separate line without any additional text. Only provide the answer, and ensure each question is answered fully.
    """

    # Step 6: Create the prompt using the template
    prompt = ChatPromptTemplate.from_template(chat_template)

    # Step 7: Set up the generative model (Google Gemini 1.5 Pro)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
    output_parser = StrOutputParser()

    # Step 8: Set up the RAG chain combining retrieval and response generation
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )

    # Step 9: Invoke the chain with the prompt (combined questions)
    output = rag_chain.invoke("")
    output_lines = output.strip().splitlines()

    # Step 10: Define default values for the answers
    sla_name = "I don't know."
    parties_involved = "I don't know."
    system_concerned = "I don't know."
    description = "I don't know."
    associated_metrics = "I don't know."
    page_number = "I don't know."

    # Step 11: Ensure that output parsing matches the structure of the response
    if len(output_lines) >= 6:
        sla_name = output_lines[0].strip() if output_lines[0].strip() else sla_name
        parties_involved = output_lines[1].strip() if output_lines[1].strip() else parties_involved
        system_concerned = output_lines[2].strip() if output_lines[2].strip() else system_concerned
        description = output_lines[3].strip() if output_lines[3].strip() else description
        associated_metrics = output_lines[4].strip() if output_lines[4].strip() else associated_metrics
        page_number = output_lines[5].strip() if output_lines[5].strip() else page_number

    # Step 12: Refine the output dynamically based on the parsed answers
    sla_info_refined = {
        "SLA Name": sla_name,
        "Parties Involved": parties_involved,
        "System Concerned": system_concerned.split("\n") if system_concerned else ["Unknown System"],
        "Description": description,
        "Associated Metrics": associated_metrics.split("\n") if associated_metrics else ["No metrics found"],
        "Page Number": page_number.split("\n") if page_number else ["No page number found"]
    }

    # Step 13: Print the refined result in JSON format
    return json.dumps(sla_info_refined, indent=4)
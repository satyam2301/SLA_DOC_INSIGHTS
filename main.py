import streamlit as st
import json

from chatbot import generate_chat_answer
from pdf_extractor import extract_text_and_tables_from_pdf
from sla_info import generate_sla_info
from pypdf_extractor import extract_text

# Streamlit app
def main():
    # Set up the title and main layout
    st.title("SLA Generation and ChatBot")

    # Sidebar for PDF Upload
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    # Button to process PDFs
    process_button = st.sidebar.button("Process PDFs")

    # Store the extracted text in session state if not already initialized
    if 'extracted_texts' not in st.session_state:
        st.session_state.extracted_texts = {}

    # If the button is clicked and PDFs are uploaded
    if process_button and uploaded_files:
        # Create a dictionary to store extracted text for each PDF
        extracted_texts = {}

        for uploaded_file in uploaded_files:
            st.sidebar.write(f"Processing file: {uploaded_file.name}...")

            # Extract text from each PDF
            # when we use azure document intelligence service 
            # pdf_text = extract_text_and_tables_from_pdf(uploaded_file)
            pdf_text = extract_text(uploaded_file)
            extracted_texts[uploaded_file.name] = pdf_text

        # Save extracted texts to session state
        st.session_state.extracted_texts = extracted_texts

    # Display the uploaded PDF names as pills in the main section
    if st.session_state.extracted_texts:
        st.write("### Select a PDF to process:")

        # Create pills for each PDF
        pdf_names = list(st.session_state.extracted_texts.keys())
        selected_pdf_name = st.pills(
            "Choose a PDF",
            options=range(len(pdf_names)),
            format_func=lambda option: pdf_names[option],
            selection_mode="single",
        )

        # Display SLA information when a PDF is selected
        if selected_pdf_name is not None:
            selected_pdf = pdf_names[selected_pdf_name]
            doc_text = st.session_state.extracted_texts[selected_pdf]

            # Button for SLA Information
            if st.button("Generate SLA Information"):
                sla_info = generate_sla_info(doc_text)
                st.json(sla_info)
                # Create a download button for SLA information as a JSON file
                json_sla_info = json.dumps(sla_info, indent=4)

                st.download_button(
                    label="Download SLA Information (JSON)",
                    data=json_sla_info,
                    file_name=f"{selected_pdf}_sla_info.json",
                    mime="application/json",
                    type="primary"
                )
            # Display conversation history for the chatbot
            st.write(f"### Conversation for {selected_pdf}")
            if 'conversation' not in st.session_state.get(selected_pdf, {}):
                st.session_state[selected_pdf] = {'conversation': []}

            # Show existing conversation history
            for idx, (question, answer) in enumerate(st.session_state[selected_pdf]['conversation']):
                st.write(f"**Q{idx + 1}:** {question}")
                st.write(f"**A{idx + 1}:** {answer}")

            # Input field to ask more questions
            question = st.text_input(f"Ask a question about '{selected_pdf}':")

            if question:
                # Generate the answer for the question asked by the user
                answer = generate_chat_answer(st.session_state.extracted_texts[selected_pdf], question)

                # Append the question-answer pair to the conversation history
                st.session_state[selected_pdf]['conversation'].append((question, answer))

                # Display the answer immediately
                st.write(f"**Answer:** {answer}")
        else:
            st.write("Please select a PDF to start the conversation.")

    else:
        if uploaded_files:
            st.write("Click the 'Process PDFs' button to start processing the PDFs.")
        else:
            st.write("Please upload one or more PDFs to proceed.")


if __name__ == "__main__":
    main()

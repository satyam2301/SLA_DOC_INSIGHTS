
from langchain.document_loaders import PyPDFLoader
import tempfile

def extract_text(uploaded_file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    document = loader.load()
    
    # You can return just the content if needed
    return "\n".join([doc.page_content for doc in document])
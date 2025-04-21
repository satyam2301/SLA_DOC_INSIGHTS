from io import BytesIO
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import os
load_dotenv()

AZURE_DOC_INTELLI_ENDPOINT = os.getenv("AZURE_DOC_INTELLI_ENDPOINT")
os.environ["AZURE_DOC_INTELLI_ENDPOINT"] = AZURE_DOC_INTELLI_ENDPOINT
AZURE_KEY=os.getenv("AZURE_KEY")
os.environ["AZURE_KEY"]=AZURE_KEY

def extract_text_and_tables_from_pdf(uploaded_file):
    # Function to check if word is within the span of a line
    def _in_span(w, spans):
        for span in spans:
            if w.span.offset >= span.offset and (w.span.offset + w.span.length) <= (span.offset + span.length):
                return True
        return False

    # Initialize the Document Intelligence client
    document_intelligence_client = DocumentIntelligenceClient(endpoint=AZURE_DOC_INTELLI_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

    # Read the content of the uploaded file into memory using BytesIO
    file_bytes = uploaded_file.read()
    file_stream = BytesIO(file_bytes)

    # Open the document and begin the analysis using the file stream
    poller = document_intelligence_client.begin_analyze_document("prebuilt-layout", body=file_stream)
    result: AnalyzeResult = poller.result()

    # Initialize a list to store extracted text and table data
    document_text = []

    # Extract text from the document (pages and lines)
    for page in result.pages:
        page_text = []

        # Process lines and words in the page
        if page.lines:
            for line in page.lines:
                words = []
                if page.words:
                    for word in page.words:
                        # Only include the word if it's within the line's span
                        if _in_span(word, line.spans):
                            words.append(word)
                # Combine words into a full line and append
                line_text = " ".join([word.content for word in words])
                page_text.append(line_text)

        # Append page-level text (lines combined into page text)
        if page_text:
            document_text.append("\n".join(page_text))

    # Extract tables and combine them with text
    combined_output = []

    # Extract tables and add them to the output
    if result.tables:
        for table_idx, table in enumerate(result.tables):
            combined_output.append(f"\n---- Table #{table_idx + 1} ----")

            # Initialize table data as a list of lists
            table_data = [['' for _ in range(table.column_count)] for _ in range(table.row_count)]

            # Fill in table data from the cells
            for cell in table.cells:
                row_idx = cell.row_index
                col_idx = cell.column_index
                table_data[row_idx][col_idx] = cell.content

            # Add the table to the combined output
            for row in table_data:
                combined_output.append("\t".join(row))

    # Add text from pages
    if document_text:
        combined_output.append("\n---- Extracted Text ----")
        combined_output.append("\n\n".join(document_text))

    # Combine all output into a single string
    doc_text = "\n".join(document_text)

    return doc_text
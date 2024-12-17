import os
import fitz  # PyMuPDF for PDF extraction
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# 1. Extract tables and text from PDFs
def extract_text_and_tables(pdf_path):
    """
    Extracts text from a PDF.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        list: List of text chunks (paragraphs).
    """
    extracted_chunks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            extracted_chunks.append(text)
    return extracted_chunks

# 2. Chunk text for embedding
def chunk_texts(texts, chunk_size=500):
    """
    Splits texts into logical chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    return splitter.split_text("\n".join(texts))

# 3. Generate embeddings using HuggingFace models
def generate_embeddings(chunks):
    """
    Embeds text chunks using HuggingFace model.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model, embeddings_model.embed_documents(chunks)

# 4. Store embeddings using FAISS
def store_embeddings(chunks, embeddings_model, storage_path="faiss_index"):
    """
    Stores embeddings in FAISS index.
    """
    os.makedirs(storage_path, exist_ok=True)
    db = FAISS.from_texts(chunks, embeddings_model)
    db.save_local(storage_path)
    print(f"Embeddings stored successfully in '{storage_path}'")
    return db

# 5. Query function for RAG with post-processing
def query_index(query, db):
    """
    Query FAISS index and return the most relevant chunk with extracted values.
    """
    results = db.similarity_search(query)
    print("\nTop Matches:")
    for i, result in enumerate(results):
        # Search for a specific year and numeric GDP value
        match = re.search(r"2015.*?Manufacturing.*?(\d{6,})", result.page_content, re.IGNORECASE)
        if match:
            print(f"\nAnswer: 2015 GDP for Manufacturing: ${match.group(1)}")
            return
        else:
            print(f"{i+1}. {result.page_content[:300]}...")  # Print partial content for debugging
    print("\nNo specific answer found. Please refine your query.")

# Main Execution Pipeline
if __name__ == "__main__":
    # Paths
    pdf_folder = "./pdfs"  # Folder containing PDF files
    storage_path = "./faiss_index"

    # Step 1: Extract and process PDFs
    if not os.path.exists(pdf_folder):
        print(f"Error: '{pdf_folder}' does not exist. Create the folder and add PDF files.")
        exit()

    # Combine data from all PDFs
    extracted_data = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            print(f"Processing file: {file_name}")
            pdf_path = os.path.join(pdf_folder, file_name)
            extracted_data += extract_text_and_tables(pdf_path)

    # Step 2: Chunk the data
    text_chunks = chunk_texts(extracted_data)
    print(f"Segmented into {len(text_chunks)} chunks.")

    # Step 3: Generate and store embeddings
    embeddings_model, _ = generate_embeddings(text_chunks)
    db = store_embeddings(text_chunks, embeddings_model, storage_path)

    print("\nPDF processing completed. Ready to accept queries!")

    # Step 4: Query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting the program.")
            break
        query_index(query, db)

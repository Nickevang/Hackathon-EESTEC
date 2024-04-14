import requests
import os
import PyPDF2
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma


# URL of the PDF to be downloaded
url = "https://www.btg-bestellservice.de/pdf/80201000.pdf"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Create a directory to store downloaded PDFs if it doesn't exist
    if not os.path.exists('pdf_files'):
        os.makedirs('pdf_files')
    
    # Extract the filename from the URL
    filename = url.split('/')[-1]

    # Specify the path to save the PDF file
    filepath = os.path.join('pdf_files', filename)

    # Write the content to a PDF file
    with open(filepath, 'wb') as pdf_file:
        pdf_file.write(response.content)
    
    print("PDF file downloaded successfully.")

    
    # Extract text from the PDF
    with open(filepath, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        print("Extracted text from PDF:")
        print(text)
      
        # Save the extracted text to a .txt file
        txt_filename = filename.split('.')[0] + '.txt'
        txt_filepath = os.path.join('pdf_files', txt_filename)
        
        print("Writing extracted text to:", txt_filepath)
        with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

       # Open the .txt file for reading
    with open(txt_filepath, 'r', encoding='utf-8') as txt_file:
        # Read the file in chunks of 100 characters
        chunk_size = 100
        chunk_number = 1
        while True:
            # Read the next chunk
            chunk = txt_file.read(chunk_size)
            # Check if the chunk is empty, indicating end of file
            if not chunk:
                break
            # Specify the filename for the chunk
            chunk_filename = f"{filename.split('.')[0]}_chunk{chunk_number}.txt"
            chunk_filepath = os.path.join('pdf_files', chunk_filename)
            # Write the chunk content to a new .txt file
            with open(chunk_filepath, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)
            print(f"Chunk {chunk_number} saved to {chunk_filepath}")
            # Increment the chunk number for the next iteration
            chunk_number += 1


            
            print(f"Extracted text saved to {txt_filepath}")
else:
    print("Failed to download PDF. Status code:", response.status_code)





client = chromadb.Client()
collection = client.get_or_create_collection(name = "my_collection")
# Iterate through each text chunk filec

"""for filename in os.listdir('pdf_files'):
    if filename.startswith(filename.split('.')[0] + '_chunk'):
        chunk_filepath = os.path.join('pdf_files', filename)
        # Read the content of the chunk file
        with open(chunk_filepath, 'r', encoding='utf-8') as chunk_file:
            chunk_content = chunk_file.read()
        # Append the content to the documents list
        documents.append(chunk_content)
        # Generate an id for the chunk (you can use filename as id)
        ids.append(filename)"""





# Define the directory containing the PDF files
directory = "pdf_files/"

# Define the prefix and suffix for the file names
prefix = "80201000_chunk"
suffix = ".txt"

# Define the range of IDs
start_id = 1
end_id = 2046

# Create an empty list to store embedded text vectors
embedded_vectors = []

# Iterate over the range of IDs and retrieve the embedded vectors for each document
for i in range(start_id, end_id + 1):
    # Generate the file path for the current ID
    file_path = os.path.join(directory, f"{prefix}{i}{suffix}")

    # Read the document content from the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        document_content = file.read()

    # Assuming you have some method to extract embeddings from document content
    # Here, we just use a placeholder method `extract_embedding`
    embedding = embedding_functions.DefaultEmbeddingFunction()



    # Append the embedded vector to the list
    embedded_vectors.append(embedding)

# Convert the list of embedded vectors to a NumPy array
embedded_vectors = np.array(embedded_vectors)

# Compute cosine similarity between pairs of embedded vectors
for i in range(start_id, end_id):
    for j in range(i + 1, end_id + 1):
        similarity = cosine_similarity([embedded_vectors[i-1]], [embedded_vectors[j-1]])[0, 0]
        print(f"Similarity between document {i} and document {j}: {similarity}")
from pathlib import Path
import time
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)
time.sleep(1)

# load documents from local directory
kb_dir = "data"
print(f"Loading data from {kb_dir}")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=80,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

documents = []
for file in Path(kb_dir).glob("*.txt"):
    loader = TextLoader(file)
    documents.extend(loader.load_and_split(text_splitter))
print(len(documents))
print(documents)


# create embeddings for all the chunks and store them in vector database
embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)

# create FAISS vector DB
vector_db = FAISS.from_documents(
    documents=documents,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)
vector_db_dir = "data/semantic-search/index/faiss"
# save vector DB to local directory
vector_db.save_local(folder_path=vector_db_dir)

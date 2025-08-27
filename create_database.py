import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

embedding_function = HuggingFaceEmbeddings(model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")  # Embedding function wrapper -> Retriever (?)


# Carrega os documentos pra dentro do programa
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents


# Divisão do texto em chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


# Transformando chunks em embeddings e salvando no Indexer
def save_to_vectordatabase(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):  # Limpa a DB antes
        shutil.rmtree(CHROMA_PATH)

    # Cria o DB e passa os embeddings para serem salvos
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_vectordatabase(chunks)


if __name__ == '__main__':
    main()

import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"


def main():

    # Criando uma CLI para inserir o input como query para o RAG
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text


    # Preparando a BD carregando o Embedding Function Wrapper pra atuar como Retriever
    embedding_function = HuggingFaceEmbeddings(model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieval
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unrelevant matches.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)

if __name__ == '__main__':
    main()
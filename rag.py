import yaml
import os

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser

import chromadb

def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def main():
    llm = ChatOllama(model='llama3', temperature=0)

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    loader = WebBaseLoader(urls)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    persist_directory = "./chroma_db"

    db = None

    if os.path.exists(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splits, embeddings, persist_directory=persist_directory)

    # 하나만
    retriever = db.as_retriever(search_kwargs={'k': 1})

    found_chunks = retriever.invoke("agent memory")

    chain = (
    llm
    | StrOutputParser())

    chain.invoke(f"{}", found_chunks)

if __name__ == "__main__":
    config = load_config('config.yaml')
    os.environ['OPENAI_API_KEY'] = config["api_key"]

    main()
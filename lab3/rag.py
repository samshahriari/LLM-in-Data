from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from documents import urls

import os


def indexing():
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(docs_list)

    if os.path.exists("./persistent_vector_store"):
        vectorstore = SKLearnVectorStore(
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
            persist_path="./persistent_vector_store"
        )
    else:
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
            persist_path="./persistent_vector_store"
        )
    vectorstore.persist()
    return vectorstore.as_retriever(k=4)


retriever = indexing()

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 

    Use the following documents to answer the question. 

    If you don't know the answer, just say that you don't know. 

    Use three sentences maximum and keep the answer concise:
    Question: {utterance} 
    Documents: {documents} 
    Answer: 
    """,
    input_variables=["utterance", "documents"],
)

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

rag_chain = prompt | llm | StrOutputParser()


while True:
    utterance = input(">")
    if utterance == "quit":
        break

    documents = retriever.invoke(utterance)

    response = rag_chain.invoke(
        {"utterance": utterance,
         "documents": documents}
    )
    print(f"{response}")

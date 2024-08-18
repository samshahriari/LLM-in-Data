
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from db_course import homework_questions, lab_questions, admin_questions, training_set, test_set

embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
v1 = embedding_model.embed_query(test_set[0])

vectorstore = SKLearnVectorStore.from_texts(
    texts=training_set,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
)

retriever = vectorstore.as_retriever(k=5)
correct = 0
for test in test_set:
    hw = False
    lab = False
    admin = False
    if test in homework_questions:
        hw = True
    elif test in lab_questions:
        lab = True
    else:
        admin = True
    documents = retriever.invoke(test)
    score = 0
    n = 0
    for doc in documents:
        n += 1
        if hw and doc.page_content in homework_questions:
            score += 1
        if lab and doc.page_content in lab_questions:
            score += 1
        if admin and doc.page_content in admin_questions:
            score += 1

    print(f"{test}:{score/n}")
    if score/n > 0.5:
        correct += 1

print(f"{correct} out of {len(test_set)}")

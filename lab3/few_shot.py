from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from db_course import homework_questions, lab_questions, admin_questions, training_set, test_set

prompt = PromptTemplate(
    template="""The following are examples of accurate classification into one of the three categories: 
    homework question, lab question and course admin question \n\n {examples} \n\n
    What should {question} classify as? Answer directly, homework, lab or admin""",
    input_variables=["examples", "question"],
)

llm = OllamaLLM(
    model="llama3.1",
    temperature=0,
)

fewshot_chain = prompt | llm | StrOutputParser()

examples = ""
for i in range(0, 10):
    examples += f"'{training_set[i]}' is a {'homework' if training_set[i] in homework_questions else 'lab' if training_set[i] in lab_questions else 'admin'} question.\n"

correct = 0
for i in range(0, len(test_set)):
    hw = False
    lab = False
    admin = False
    if test_set[i] in homework_questions:
        hw = True
    elif test_set[i] in lab_questions:
        lab = True
    else:
        admin = True
    response = fewshot_chain.invoke(
        {"examples": examples, "question": test_set[i]}
    )

    if hw and "homework" in response.lower():
        correct += 1
    if lab and "lab" in response.lower():
        correct += 1
    if admin and "admin" in response.lower():
        correct += 1

    print(f"{test_set[i]}:{response}")

print(correct)

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = PromptTemplate(
    # template="""{history}\n{utterance}""",
    template="""<Start of old chat history>\n{history}\n<end of old chat history>\n{utterance}""",

    input_variables=["history", "utterance"],
)
llm = OllamaLLM(
    model="llama3.1",
    temperature=0,
)
turn_chain = prompt | llm | StrOutputParser()
history = ""
while True:
    utterance = input(">")
    if utterance == "quit":
        break
    response = turn_chain.invoke(
        {"history": history,
         "utterance": utterance}
    )
    print(f"{response}")
    # history += history + "\n" + response
    history = history + "\n" + utterance+"\n" + response + "\n"

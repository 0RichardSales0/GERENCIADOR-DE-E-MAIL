import streamlit as st
from langchain.vectorstores import FAISS 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

load_dotenv()

loader = CSVLoader(file_path = "copia.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

llm = ChatOpenAI(temperature=0, model="chatgpt-4o-latest")

template = """
você é um assistente virtual de uma empresa que 
trabalha de suporte Atendimento ao cliente de TI N1, sua função é responder e-mail que são enviados com frequencia, você trabalha empresa chamada teste, 
crie um script de e-mail se identificando como suporte e descrevendo sobre a empresa,não utilize topicos e seja mas sucinto, não 
precisa do nome e numero, seu nome é Richad 

Escreva melhor resposta que deveria dar para esse cliente:
"""

prompt = PromptTemplate(
    input_variables = ["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

generate_response("Meu computador esta muito lento. Como posso resolver isso?")

def main():
    st.set_page_config(
        page_title="E-mail manager", page_icon=":bird:")
    st.header("E-mail manager")
    message = st.text_area("E-mail do cliente")

    if message:
        st.write("Gerando em E-mail resposta baseado nas melhores práticas...")

        result = generate_response(message)

        st.info(result)

if __name__ == '__main__':
    main()


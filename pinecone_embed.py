import os
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Inicializar Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Nombre del índice en Pinecone
index_name = "integrationai"

# Modelo de embeddings de OpenAI
model_name = "text-embedding-ada-002"

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Cargar el índice en Pinecone con los embeddings correctos
vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embed)

# Definir la plantilla de pregunta para el chatbot
def get_answer(query):
  question = query
  template = """El usuario está interesado en saber más sobre la asignación del espectro radioeléctrico en Colombia.
Responde únicamente preguntas relacionadas con el espectro radioeléctrico en Colombia y con la información contenida en la base de datos vectorial. 
Las preguntas externas a este tema deben recibir una respuesta indicando que no puedes responder.
Usa los siguientes fragmentos de contexto para responder la pregunta de manera concisa e ignora los signos de interrogación:
  {context}
  Question: {question}
  Helpful Answer:"""
  QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

  llm = ChatOpenAI(model_name="gpt-4", temperature=0.1)
  qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
  )
  result = qa_chain({"query": question})
  return result["result"]
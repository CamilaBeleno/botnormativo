import os
import json
import io
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 🔹 Cargar las credenciales de Pinecone y OpenAI desde el entorno
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "integrationai"  # Mismo índice que en pinecone_embed.py

# 🔹 Inicializar conexión a Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# 🔹 Configurar embeddings
embed = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

# 🔹 Conectar al índice de Pinecone
vector_store = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embed)

# 🔹 Configuración de Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

FOLDER_ID = "1wFvhqW_ujNRk-kqDW8uxF4fMGfyXEB4E"  # 🔹 Reemplázalo con el ID de tu carpeta de Google Drive

# 🔹 RUTA DEL ARCHIVO DE METADATOS (para evitar procesar documentos repetidos)
METADATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "documentos_procesados.json")


def cargar_metadatos():
    """Carga los documentos que ya han sido procesados."""
    if not os.path.exists(METADATA_FILE):
        return {}

    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def guardar_metadatos(data):
    """Guarda los documentos procesados en un archivo JSON."""
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def obtener_documentos_drive():
    """Obtiene la lista de documentos en Google Drive sin descargarlos."""
    query = f"'{FOLDER_ID}' in parents and mimeType='application/pdf'"
    file_list = drive.ListFile({'q': query}).GetList()

    return {file['title']: file['id'] for file in file_list}


def contar_paginas(file_id):
    """Cuenta cuántas páginas tiene un PDF directamente desde Google Drive."""
    file = drive.CreateFile({'id': file_id})
    file_content = io.BytesIO(file.GetContentBinary())

    pdf_reader = PdfReader(file_content)
    return len(pdf_reader.pages)


def fragmentar_documento(file_id):
    """Lee y fragmenta un PDF directamente desde Google Drive sin guardarlo."""
    file = drive.CreateFile({'id': file_id})
    file_content = io.BytesIO(file.GetContentBinary())

    loader = PyPDFLoader(file_content)
    documentos = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documentos)


def actualizar_pinecone(fragmentos, doc_name):
    """Envía los fragmentos a Pinecone para actualizar el índice."""
    for i, frag in enumerate(fragmentos):
        vector_store.add_texts([frag.page_content], metadatas=[{"source": doc_name}])


if __name__ == "__main__":
    print("Iniciando actualización de documentos...")

    # Cargar metadatos
    documentos_procesados = cargar_metadatos()
    documentos_drive = obtener_documentos_drive()

    for doc, file_id in documentos_drive.items():
        if doc in documentos_procesados:
            print(f"{doc} ya fue procesado. Omitiendo...")
            continue

        print(f"Procesando: {doc}")

        num_paginas = contar_paginas(file_id)
        fragmentos = fragmentar_documento(file_id)
        actualizar_pinecone(fragmentos, doc)

        # Guardar en metadatos
        documentos_procesados[doc] = {"id": file_id, "num_paginas": num_paginas}
        guardar_metadatos(documentos_procesados)

    print("Actualización completada.")

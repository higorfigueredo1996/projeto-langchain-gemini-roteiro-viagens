from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import time

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.5,
    api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf"
]

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ], []
)

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200
).split_documents(documentos)

TAMANHO_LOTE = 20

vectorstore = None
for i in range(0, len(pedacos), TAMANHO_LOTE):
    lote = pedacos[i:i + TAMANHO_LOTE]
    print(f"Processando lote {i // TAMANHO_LOTE + 1} ({len(lote)} pedacos)...")
    if vectorstore is None:
        vectorstore = FAISS.from_documents(lote, embeddings)
    else:
        vectorstore.add_documents(lote)
    time.sleep(20)

dados_recuperados = vectorstore.as_retriever(search_kwargs={"k":10})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({
        "query": pergunta, "contexto":contexto
    })

print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão platinum"))
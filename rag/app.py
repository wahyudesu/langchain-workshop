from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_pinecone import PineconeEmbeddings, Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv
import chainlit as cl

# Load variabel environment dari file .env (jika ada)
load_dotenv()

# Ambil API key OpenAI dari environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY tidak ditemukan di environment atau .env")


@cl.on_chat_start
async def on_chat_start():
    # Minta user upload file PDF/TXT
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Silakan upload file PDF atau TXT untuk mulai QA!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]
    msg = cl.Message(content=f"Memproses `{file.name}`...")
    await msg.send()

    # Pilih loader sesuai tipe file
    if file.type == "text/plain":
        Loader = TextLoader  # Loader untuk file teks
    elif file.type == "application/pdf":
        Loader = PyPDFLoader  # Loader untuk file PDF
    else:
        await cl.Message(content="Tipe file tidak didukung.").send()
        return

    # Load dokumen dari file
    loader = Loader(file.path)
    documents = loader.load()

    # Split dokumen menjadi potongan-potongan kecil (chunk)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Buat embeddings dari teks menggunakan OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Simpan embeddings ke vectorstore lokal (Chroma)
    vectordb = Chroma.from_documents(texts, embeddings)
    # Buat retriever untuk pencarian dokumen relevan
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    # Inisialisasi LLM (ChatOpenAI)
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    # Buat chain RetrievalQA untuk tanya jawab berbasis dokumen
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Simpan chain ke session user
    cl.user_session.set("qa", qa)
    msg.content = f"`{file.name}` siap! Silakan tanya apa saja tentang dokumen ini."
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    # Ambil chain RetrievalQA dari session
    qa = cl.user_session.get("qa")
    if qa is None:
        await cl.Message(content="Silakan upload file terlebih dahulu.").send()
        return
    msg = cl.Message(content="")
    # Jalankan chain untuk menjawab pertanyaan user
    result = qa.invoke(message.content)
    # Ambil jawaban string dari dict hasil chain
    answer = (
        result["result"] if "result" in result else result.get("answer", str(result))
    )
    await msg.stream_token(answer)
    await msg.send()

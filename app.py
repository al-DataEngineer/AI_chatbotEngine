import csv
import os
import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

KB_PATH = "knowledge_base.csv"

st.set_page_config(page_title="MinAIEngine - AI Assistant", page_icon="🤖")
st.title("🤖 MinAIEngine")
st.caption("Asisten AI berbasis Retrieval-Augmented Generation (RAG) • Dibangun oleh Ahmad Alimin")

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ API Key terhubung otomatis")
    else:
        groq_api_key = st.text_input("Masukkan Groq API Key", type="password", placeholder="gsk_...")
    
    k_docs = st.slider("Jumlah dokumen rujukan (Top-K)", min_value=1, max_value=6, value=3)
    st.markdown("---")
    st.markdown("**Arsitektur:**\n- LLM: Groq Active Model\n- Embedding: Multilingual MiniLM-L12\n- Framework: LangChain + Streamlit")

if not groq_api_key:
    st.info("👈 Masukkan **GROQ_API_KEY** di sidebar sebelah kiri untuk mulai menggunakan MinAIEngine.")
    st.stop()

@st.cache_resource
def init_vector_store():
    df = pd.read_csv(KB_PATH)
    docs = []
    for _, row in df.iterrows():
        konten = f"Kategori: {row['kategori']}\nPertanyaan: {row['pertanyaan']}\nJawaban: {row['jawaban']}"
        docs.append(Document(page_content=konten, metadata={"kategori": row["kategori"]}))
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    return vector_store

vector_store = init_vector_store()

def simpan_ke_knowledge_base(pertanyaan: str, jawaban: str, kategori: str = "Percakapan") -> bool:
    """Simpan Q&A baru ke knowledge_base.csv jika belum ada, lalu tambahkan ke vector_store."""
    q_clean = pertanyaan.strip()
    a_clean = jawaban.strip()
    if not q_clean or not a_clean:
        return False

    if os.path.exists(KB_PATH):
        df = pd.read_csv(KB_PATH)
        sudah_ada = df["pertanyaan"].astype(str).str.strip().str.lower().eq(q_clean.lower()).any()
        if sudah_ada:
            return False

    butuh_newline = False
    if os.path.exists(KB_PATH) and os.path.getsize(KB_PATH) > 0:
        with open(KB_PATH, "rb") as f:
            f.seek(-1, os.SEEK_END)
            butuh_newline = f.read(1) not in (b"\n", b"\r")

    with open(KB_PATH, mode="a", encoding="utf-8", newline="") as f:
        if butuh_newline:
            f.write("\n")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([kategori, q_clean, a_clean])

    konten = f"Kategori: {kategori}\nPertanyaan: {q_clean}\nJawaban: {a_clean}"
    vector_store.add_documents([Document(page_content=konten, metadata={"kategori": kategori})])
    return True

from groq import Groq

# Pilih model chat Groq yang stabil dan aktif
client = Groq(api_key=groq_api_key)
semua_model = [m.id for m in client.models.list().data]

# Urutan prioritas model chat yang paling stabil di Groq
prioritas_model = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "gemma2-9b-it"
]

model_chat_tersedia = [
    m for m in semua_model
    if not any(x in m.lower() for x in ["guard", "whisper", "tts", "vision", "prompt", "safeguard"])
]

kandidat_model = [m for m in prioritas_model if m in model_chat_tersedia]
for m in model_chat_tersedia:
    if m not in kandidat_model and ("llama" in m or "gemma" in m or "qwen" in m or "mixtral" in m):
        kandidat_model.append(m)

if not kandidat_model:
    kandidat_model = model_chat_tersedia or semua_model

model_aktif = kandidat_model[0]
st.sidebar.caption(f"Model aktif: `{model_aktif}`")

def buat_llm(nama_model: str) -> ChatGroq:
    return ChatGroq(
        api_key=groq_api_key,
        model=nama_model,
        temperature=0.3,
        max_tokens=800
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", """Kamu adalah MinAIEngine, asisten AI cerdas dan serba bisa yang dikembangkan oleh Ahmad Alimin, dengan keahlian khusus di bidang Statistika, Data Analyst, Data Scientist, Data Engineer, dan AI Engineering.

Panduan menjawab:
1. Gunakan "Konteks Basis Pengetahuan" di bawah ini sebagai referensi utama apabila relevan dengan pertanyaan pengguna (terutama untuk identitas MinAIEngine dan konsep inti).
2. Jika pertanyaan pengguna tidak tercantum di dalam konteks atau membutuhkan penjelasan lebih luas (seperti contoh kode, studi kasus, atau topik di luar basis pengetahuan), JANGAN menolak—tetap jawab secara akurat, lengkap, dan membantu menggunakan pengetahuanmu sendiri.
3. Perhatikan riwayat percakapan agar jawabanmu nyambung dengan pertanyaan sebelumnya.
4. Jawab dalam bahasa Indonesia yang jelas, terstruktur, dan mudah dipahami. Jika memang tidak tahu suatu fakta spesifik atau data real-time, sampaikan dengan jujur.

Konteks Basis Pengetahuan:
{context}

Riwayat Percakapan Terakhir:
{history}"""),
    ("human", "{question}")
])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya **MinAIEngine**. Ada yang bisa saya bantu hari ini?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ketik pertanyaanmu untuk MinAIEngine..."):
    history_text = "\n".join(
        f"{'Pengguna' if m['role'] == 'user' else 'MinAIEngine'}: {m['content']}"
        for m in st.session_state.messages[-6:]
    )

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("MinAIEngine sedang berpikir..."):
            retrieved_docs = vector_store.similarity_search(user_query, k=k_docs)
            context_text = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])

            answer = None
            last_error = None
            for nama_model in kandidat_model:
                try:
                    chain = prompt | buat_llm(nama_model)
                    response = chain.invoke({
                        "context": context_text,
                        "history": history_text,
                        "question": user_query
                    })
                    answer = response.content
                    break
                except Exception as e:
                    last_error = (nama_model, e)

            if answer is not None:
                st.markdown(answer)

                with st.expander("📚 Lihat Sumber Referensi yang Diambil"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Sumber {i} ({doc.metadata.get('kategori', 'Umum')}):**")
                        st.text(doc.page_content)

                if auto_save_kb:
                    if simpan_ke_knowledge_base(user_query, answer):
                        st.caption("✅ Tersimpan otomatis ke `knowledge_base.csv`")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                failed_model, err = last_error
                st.error(f"Gagal memanggil Groq API (terakhir mencoba `{failed_model}`): {err}")

# 🤖 MinAIEngine — Personal RAG AI Assistant

**MinAIEngine** adalah aplikasi asisten kecerdasan buatan berbasis arsitektur **Retrieval-Augmented Generation (RAG)** yang dibangun menggunakan Python, LangChain, Groq LLM, dan Streamlit. Sistem ini dirancang untuk menjawab pertanyaan secara akurat berdasarkan basis pengetahuan terverifikasi tanpa berhalusinasi.

## ✨ Fitur Utama
- **Semantic Search & Retrieval:** Menggunakan model embedding multibahasa (`paraphrase-multilingual-MiniLM-L12-v2`) dan `InMemoryVectorStore` untuk mencari konteks dokumen berdasarkan kemiripan makna.
- **Dynamic Model Selection:** Mendeteksi otomatis model LLM aktif pada API Groq untuk menjamin ketersediaan layanan.
- **Anti-Hallucination Guardrails:** Menggunakan *system prompt* ketat dan *temperature* rendah agar AI hanya menjawab berdasarkan fakta dokumen dan jujur menolak pertanyaan di luar konteks.
- **Source Transparency:** Menampilkan potongan dokumen rujukan (Top-K) yang diambil sistem di setiap jawaban.

## 🛠️ Teknologi yang Digunakan
- **Frontend:** Streamlit
- **Orchestration:** LangChain (`langchain-core`, `langchain-groq`, `langchain-huggingface`)
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM Provider:** Groq API

## 🚀 Cara Menjalankan Secara Lokal
1. Clone repositori ini:
   ```bash
   git clone [https://github.com/al-DataEngineer/MinAIEngine.git](https://github.com/al-DataEngineer/MinAIEngine.git)
   cd MinAIEngine
# ScholarSage 📚 – Smart Research Paper Assistant

ScholarSage is an AI-powered Streamlit application that helps researchers and students explore academic papers more efficiently. It allows you to:

- 🔍 Search research papers by topic using the Semantic Scholar API
- 📥 Automatically fetch the best-matching paper from arXiv
- 🧠 Summarize papers using OpenAI’s GPT (RAG-style with LangChain)
- 💬 Ask detailed questions about the paper
- 📄 Preview or download the full PDF
- 🖼 View extracted images from the paper

---

## 🚀 Features

- Built with **Streamlit**, **LangChain**, and **OpenAI API**
- Uses **vector-based RAG architecture** to provide accurate context-aware summaries and Q&A
- Extracts **inline images** from the paper for better visual comprehension
- Handles large PDFs gracefully by offering download-only previews

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/scholarsage.git
cd scholarsage
pip install -r requirements.txt

🧪 Usage
streamlit run app.py
You will be prompted to enter your OpenAI API key and a research topic. ScholarSage will handle the rest.

🧩 Dependencies
Streamlit
LangChain
OpenAI SDK
BeautifulSoup4
Requests
pdfminer.six
pymupdf (fitz)
chromadb
pandas

🔒 Notes
Make sure you have an active OpenAI API key with access to GPT-3.5-turbo or GPT-4.
Semantic Scholar’s API is used for free (no API key required).

🧠 Inspired by
Research scholars who want to automate and scale their paper review and comprehension workflows.


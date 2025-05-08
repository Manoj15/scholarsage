import os
import requests
import re
import tempfile
import streamlit as st
import base64
import time
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz  # for extract_images_from_pdf

# --- Utilities ---
def safe_get(url, params=None, retries=5, wait_time=10):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 429:
                st.warning(f"⚠️ Rate limited (429). Waiting {wait_time} seconds... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            elif response.status_code == 200:
                return response
            else:
                st.error(f"❌ Request failed with status {response.status_code} (Attempt {attempt + 1})")
                st.info(f"URL: {url}\nParams: {params}")
                time.sleep(wait_time)
        except Exception as e:
            st.error(f"❌ Exception during request: {e} (Attempt {attempt + 1})")
            st.info(f"URL: {url}\nParams: {params}")
            time.sleep(wait_time)
    raise Exception("Max retries reached.")

@st.cache_data(show_spinner=False)
def get_titles_from_semantic_scholar(topic, limit=5):
    st.info(f"🔍 Sending request to Semantic Scholar API with topic: '{topic}' and limit: {limit}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={topic}&limit={limit}&fields=title"
    response = safe_get(url)
    data = response.json()
    titles = [paper['title'] for paper in data.get('data', []) if 'title' in paper]
    return titles

def clean_title(title):
    title = re.sub(r'[^\w\s]', '', title)
    words = title.split()
    important_words = words[:8]
    return " ".join(important_words)

def find_best_arxiv_match(target_title, entries):
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    best_score = 0
    best_entry = None
    for entry in entries:
        title = entry.find('title').text.strip()
        score = similarity(target_title, title)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry

@st.cache_data(show_spinner=False)
def search_arxiv_by_keywords(title_keywords, original_title):
    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{title_keywords}", "start": 0, "max_results": 5}
    try:
        response = safe_get(base_url, params=params)
        feed = BeautifulSoup(response.text, "xml")
        entries = feed.find_all('entry')
        best_entry = find_best_arxiv_match(original_title, entries)
        if best_entry:
            arxiv_id = best_entry.find('id').text.split('/abs/')[-1]
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except Exception as e:
        st.error(f"Error searching ArXiv: {e}")
    return None

def download_pdf(pdf_url):
    try:
        st.info(f"📡 Downloading from: {pdf_url}")
        response = requests.get(pdf_url, stream=True, timeout=10)
        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
            st.info("📥 Verifying and downloading PDF...")
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            for chunk in response.iter_content(1024):
                temp_pdf.write(chunk)
            temp_pdf.close()
            st.success("✅ PDF successfully downloaded!")
            return temp_pdf.name
        else:
            st.error("❌ Invalid PDF or download error.")
    except Exception as e:
        st.error(f"Error downloading PDF: {e}")
    return None

def analyze_with_rag(pdf_path, openai_key, task_type, question=None):
    os.environ["OPENAI_API_KEY"] = openai_key

    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 15})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="refine")

    if task_type == "Summarise":
        prompt = (
            "You are a post-doctoral researcher who reviews and understands research papers. You summarise the given document in terms of the necessary technical concepts to make notes to write your next research paper. You can generate the summary as long as possible to cover all the technical details in the document. Please make the notes with headings, sub-headings, bullet points and paragraphs. Return the text in markdown format."
        )
    else:
        prompt = question

    return rag_chain.run(prompt)

def extract_images_from_pdf(pdf_path, max_images=5):
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_ext}").name
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                image_paths.append(img_path)
                if len(image_paths) >= max_images:
                    return image_paths
    except Exception as e:
        st.warning(f"Could not extract images: {e}")
    return image_paths

# --- Streamlit UI ---
st.set_page_config(page_title="ScholarSage - Research Assistant", layout="wide")
st.title("📚 ScholarSage: Smart Research Paper Assistant")

if st.button("🧹 Clear All Cached Data"):
    st.cache_data.clear()
    st.success("🔄 Cache cleared!")

with st.sidebar:
    openai_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
    if openai_key:
        st.success("API Key Set ✅")

st.markdown("### 🔍 Search for Research Papers")
topic = st.text_input("Enter a research topic")
limit = st.slider("Number of paper titles to fetch", 1, 10, 5)
pdf_url = None
pdf_path = None

if topic and openai_key:
    with st.spinner("🔍 Searching Semantic Scholar for papers..."):
        titles = get_titles_from_semantic_scholar(topic, limit)
    st.markdown("#### Select a paper to analyze")
    selected_title = st.radio("Available Papers", titles, key="selected_title")

    if selected_title:
        title_keywords = clean_title(selected_title)
        pdf_url = search_arxiv_by_keywords(title_keywords, selected_title)

        if pdf_url and st.button("📥 Download & Load Paper"):
            with st.spinner("⏳ Downloading and verifying PDF from ArXiv..."):
                pdf_path = download_pdf(pdf_url)
                if pdf_path:
                    st.session_state['pdf_path'] = pdf_path

    if st.session_state.get('pdf_path'):
        pdf_path = st.session_state['pdf_path']

        if os.path.exists(pdf_path):
            st.markdown(f"✅ Showing results for: **{selected_title}**")

            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name=os.path.basename(pdf_path))

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.text(f"PDF size: {os.path.getsize(pdf_path)} bytes")

            if os.path.getsize(pdf_path) > 3_500_000:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.warning("📄 PDF is too large to preview inline. You can download and view it externally:")
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF", f, file_name=os.path.basename(pdf_path))
            else:
                col1, col2 = st.columns([2, 1])
                base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                pdf_display = f"""
                    <iframe src=\"data:application/pdf;base64,{base64_pdf}\" width=\"100%\" height=\"600\" type=\"application/pdf\"></iframe>
                """
                with col1:
                    st.markdown(pdf_display, unsafe_allow_html=True)
            with col2:
                user_confirmed = st.radio("Is this the correct paper?", ["Yes", "No"], horizontal=False)
                if user_confirmed == "No":
                    alt_pdf_url = st.text_input("Paste the correct ArXiv PDF link")
                    if alt_pdf_url and st.button("🔁 Load Alternative PDF"):
                        with st.spinner("📥 Downloading and verifying new PDF..."):
                            new_pdf_path = download_pdf(alt_pdf_url)
                            if new_pdf_path and os.path.exists(new_pdf_path):
                                st.session_state['pdf_path'] = new_pdf_path
                                st.success("✅ New PDF loaded successfully!")
                                with open(new_pdf_path, "rb") as f:
                                    new_pdf_bytes = f.read()
                                st.text(f"PDF size: {os.path.getsize(new_pdf_path)} bytes")
                                base64_new_pdf = base64.b64encode(new_pdf_bytes).decode("utf-8")
                                new_pdf_display = f"""
                                    <iframe src=\"data:application/pdf;base64,{base64_new_pdf}\" width=\"100%\" height=\"600\" type=\"application/pdf\"></iframe>
                                """
                                with col1:
                                    st.markdown(new_pdf_display, unsafe_allow_html=True)
                                pdf_path = new_pdf_path

            task = st.radio("📌 What do you want to do?", ["Summarise", "Ask a Question"])
            user_q = st.text_input("💬 Ask a custom question") if task == "Ask a Question" else None

            if st.button("Run Analysis"):
                with st.spinner("🤖 Processing document and generating response with GPT..."):
                    result = analyze_with_rag(pdf_path, openai_key, task, user_q)
                    st.markdown(result, unsafe_allow_html=True)

                    st.markdown("## 📸 Images Extracted from the Paper")
                    image_paths = extract_images_from_pdf(pdf_path)
                    for img_path in image_paths:
                        st.image(img_path, width=500)
        else:
            st.error("❌ PDF file not found. Please download again.")
else:
    st.info("🔐 Please enter your OpenAI API key and search topic to begin.")

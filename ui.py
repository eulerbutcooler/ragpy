import uuid

import boto3
import requests
import streamlit as st

# Configuration
API_URL = "http://localhost:8000/api/v1"
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS = "minioadmin"
MINIO_SECRET = "minioadmin"
BUCKET_NAME = "course-documents"
COURSE_ID = "course_test_api"
HEADERS = {"x-internal-token": "development_secret_token"}

st.set_page_config(page_title="RAG Chat & Quiz", page_icon="🤖", layout="wide")
st.title("📚 RAG Chat & Quiz Interface")

# Setup S3 connection
@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3", 
        endpoint_url=MINIO_ENDPOINT, 
        aws_access_key_id=MINIO_ACCESS, 
        aws_secret_access_key=MINIO_SECRET
    )

s3 = get_s3_client()

# Ensure bucket exists
try:
    s3.head_bucket(Bucket=BUCKET_NAME)
except:
    s3.create_bucket(Bucket=BUCKET_NAME)

tabs = st.tabs(["📝 Upload & Chat", "🧠 Generate Quiz"])

# --- TAB 1: UPLOAD & CHAT ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader("Upload course material", type=["txt", "pdf", "md", "docx", "pptx"])

        if uploaded_file and st.button("📤 Ingest Document"):
            file_id = str(uuid.uuid4())
            file_name = uploaded_file.name
            key = f"{COURSE_ID}/{file_id}_{file_name}"
            
            with st.spinner("Uploading to MinIO..."):
                s3.upload_fileobj(uploaded_file, BUCKET_NAME, key)
            
            with st.spinner("Queueing for ingestion..."):
                res = requests.post(f"{API_URL}/ingest/", headers=HEADERS, json={
                    "bucket": BUCKET_NAME, 
                    "key": key, 
                    "course_id": COURSE_ID,
                    "file_id": file_id, 
                    "file_name": file_name, 
                    "teacher_id": "teacher_1"
                })
                if res.status_code == 202:
                    st.success("✅ Uploaded and queued! Wait a moment for processing.")
                else:
                    st.error(f"Failed to ingest: {res.text}")

    with col2:
        st.subheader("2. Chat with Documents")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render chat messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # TTS Button for assistant messages
                if msg["role"] == "assistant":
                    if st.button("🔊 Read Aloud", key=f"tts_{i}"):
                        with st.spinner("Generating speech..."):
                            tts_res = requests.post(f"{API_URL}/audio/speak", headers=HEADERS, json={"text": msg["content"]})
                            if tts_res.ok:
                                st.audio(tts_res.content, format="audio/wav", autoplay=True)
                            else:
                                st.error("Failed to generate audio.")
                
                if msg.get("citations"):
                    with st.expander("Sources"):
                        for cite in msg["citations"]:
                            st.write(f"- {cite['file_name']} (Score: {cite.get('score', 'N/A')})")

        # Allow text OR audio input
        final_prompt = None
        
        text_prompt = st.chat_input("Ask a question about the document...")
        if text_prompt:
            final_prompt = text_prompt
            
        audio_val = st.audio_input("Or speak your question")
        if audio_val:
            audio_hash = hash(audio_val.getvalue())
            if st.session_state.get("last_audio_hash") != audio_hash:
                with st.spinner("Transcribing..."):
                    files = {"file": ("audio.wav", audio_val.getvalue(), "audio/wav")}
                    res = requests.post(f"{API_URL}/audio/transcribe", headers=HEADERS, files=files)
                    if res.ok:
                        final_prompt = res.json().get("text", "")
                        st.session_state["last_audio_hash"] = audio_hash
                    else:
                        st.error("Transcription Failed")

        if final_prompt:
            st.session_state.messages.append({"role": "user", "content": final_prompt})

            # Build conversation history for the API (last 10 messages, excluding the current one)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-10:]
            ]

            with st.spinner("Thinking..."):
                resp = requests.post(f"{API_URL}/chat/", headers=HEADERS, json={
                    "course_id": COURSE_ID,
                    "query": final_prompt,
                    "stream": False,
                    "history": history,
                })
                
                if resp.ok:
                    data = resp.json()
                    reply = data["answer"]
                    citations = data.get("citations", [])
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reply,
                        "citations": citations
                    })
                    st.rerun()
                else:
                    st.error("RAG Engine error!")



# --- TAB 2: GENERATE QUIZ ---
with tabs[1]:
    st.subheader("Quiz Generator")
    num_chunks = st.slider("Max Context Chunks to analyze", min_value=5, max_value=100, value=20)
    
    if st.button("✨ Generate Quiz"):
        with st.spinner("Analyzing documents and creating quiz..."):
            res = requests.post(f"{API_URL}/quiz/generate", headers=HEADERS, json={
                "course_id": COURSE_ID,
                "limit_chunks": num_chunks
            })
            
            if res.ok:
                quiz_data = res.json()
                questions = quiz_data.get("questions", [])
                
                st.success(f"Generated {len(questions)} questions for Course {COURSE_ID}!")
                
                for i, q in enumerate(questions, 1):
                    with st.container():
                        st.markdown(f"**Q{i}: {q['question']}**")
                        
                        if q["type"] == "mcq" and q.get("choices"):
                            for choice in q["choices"]:
                                st.write(f"- **{choice['label']}**: {choice['text']}")
                                
                        with st.expander("Show Answer"):
                            st.info(f"**Answer:** {q['answer']}")
                        st.divider()
            else:
                if res.status_code == 404:
                    st.warning("No context found! Make sure you upload and ingest documents first.")
                else:
                    st.error(f"Failed to generate quiz: {res.text}")

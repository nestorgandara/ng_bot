import os, json, boto3, faiss, numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
from base64 import b64encode

st.set_page_config(page_title="Nestor’s Bot", page_icon="🤖", layout="centered")
load_dotenv()

# ==============================================================
# IMAGE LOAD FIX (absolute, reliable path)
# ==============================================================
def get_base64_image(image_filename):
    for path in [".", "images", "static"]:
        full_path = os.path.join(path, image_filename)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                return b64encode(f.read()).decode()
    return ""

bot_image = get_base64_image("bot_app.png")

# ==============================================================
# CSS: Vision-Style with proper single reflection effect
# ==============================================================
st.markdown(f"""
<style>
body {{
  margin: 0;
  overflow-x: hidden;
  font-family: "SF Pro Display", "Helvetica Neue", sans-serif;
  background: radial-gradient(circle at 50% 20%, #001f3f, #000814);
  color: white;
}}

div.block-container {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  text-align: center;
  height: 100vh;
  padding: 1rem;
}}

.hero {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  width: 100%;
  max-width: 900px;
  margin-top: 1rem;
}}

.hero-img {{
  width: 100%;
  border-radius: 25px;
  box-shadow: 0 40px 100px rgba(0,200,255,0.4);
  animation: float 6s ease-in-out infinite;
  object-fit: contain;
}}

.glow {{
  position: absolute;
  bottom: -30px;
  width: 80%;
  height: 120px;
  border-radius: 50%;
  background: radial-gradient(circle at 50% 50%, rgba(0,255,255,0.35), transparent 70%);
  filter: blur(40px);
  z-index: -1;
}}

@keyframes float {{
  0% {{ transform: translateY(0px); }}
  50% {{ transform: translateY(-12px); }}
  100% {{ transform: translateY(0px); }}
}}

h1 {{
  font-weight: 800;
  font-size: 3em;
  margin-top: 2rem;
  background: linear-gradient(90deg, #00f2ff, #0077ff, #b3f7ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 25px rgba(0,180,255,0.7);
}}

.chat-bubble {{
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 14px;
  padding: 14px 18px;
  margin-top: 10px;
  width: 80%;
  backdrop-filter: blur(15px);
  box-shadow: 0 0 20px rgba(0,150,255,0.2);
  text-align: left;
  animation: fadeIn 0.8s ease;
}}
@keyframes fadeIn {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
</style>

<div class="hero">
  <img src="data:image/png;base64,{bot_image}" class="hero-img">
  <div class="glow"></div>
</div>

<h1>Nestor’s Bot</h1>
""", unsafe_allow_html=True)

# ==============================================================
# BACKGROUND PARTICLES
# ==============================================================
st.components.v1.html("""
<canvas id="coreCanvas"></canvas>
<script>
const canvas=document.getElementById('coreCanvas');
const ctx=canvas.getContext('2d');
let w,h;
function resize(){w=canvas.width=window.innerWidth;h=canvas.height=window.innerHeight;}
window.addEventListener('resize',resize);resize();

let particles=[];
for(let i=0;i<100;i++){
  particles.push({x:Math.random()*w,y:Math.random()*h,r:Math.random()*2+0.5,
    dx:(Math.random()-0.5)*0.3,dy:(Math.random()-0.5)*0.3,color:`hsla(${Math.random()*360},70%,60%,0.8)`});
}
function draw(){
  ctx.clearRect(0,0,w,h);
  for(let p of particles){
    ctx.beginPath();
    ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx.fillStyle=p.color;
    ctx.fill();
    p.x+=p.dx;p.y+=p.dy;
    if(p.x<0)p.x=w;if(p.x>w)p.x=0;
    if(p.y<0)p.y=h;if(p.y>h)p.y=0;
  }
  requestAnimationFrame(draw);
}
draw();
</script>
""", height=0)

# ==============================================================
# AWS Bedrock Client
# ==============================================================
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# ==============================================================
# Helper Functions
# ==============================================================
def extract_text(file):
    if file.name.endswith(".pdf"):
        return "\n".join(p.extract_text() for p in PdfReader(file).pages if p.extract_text())
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    if file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

def embed_text(chunks):
    embs = []
    for ch in chunks:
        body = json.dumps({"inputText": ch})
        r = bedrock.invoke_model(modelId="amazon.titan-embed-text-v1",
                                 body=body, accept="application/json", contentType="application/json")
        embs.append(json.loads(r["body"].read())["embedding"])
    return np.array(embs).astype("float32")

def chunk_text(text, n=800):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]

def retrieve_context(q, index, texts, k=3):
    qv = embed_text([q])
    _, idx = index.search(qv, k)
    return [texts[i] for i in idx[0] if i < len(texts)]

def ask_claude(prompt, context=""):
    msg = f"Use this context if relevant.\n\nContext:\n{context}\n\nQuestion: {prompt}"
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"type": "text", "text": msg}]}],
        "max_tokens": 700,
        "anthropic_version": "bedrock-2023-05-31"
    })
    r = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=body, accept="application/json", contentType="application/json"
    )
    return json.loads(r["body"].read())["content"][0]["text"]

# ==============================================================
# Sidebar + Chat Interface
# ==============================================================
st.sidebar.header("⚙️ Settings")
summary_mode = st.sidebar.checkbox("🧾 Summary Mode", value=True)

uploaded = st.file_uploader("📎 Upload your document (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
if uploaded:
    st.info(f"📘 Processing {uploaded.name} ...")
    text = extract_text(uploaded)
    chunks = chunk_text(text)
    vecs = embed_text(chunks)
    idx = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)
    st.session_state["idx"] = idx
    st.session_state["texts"] = chunks
    st.success("✅ Document embedded successfully!")

query = st.text_area("💭 Ask your question:", "Nestor ask anything?")
if st.button("🚀 Ask Nestor"):
    with st.spinner("✨ Thinking..."):
        context = ""
        if "idx" in st.session_state:
            context = "\n\n".join(retrieve_context(query, st.session_state["idx"], st.session_state["texts"]))
        ans = ask_claude(query, context)
        st.markdown(f"<div class='chat-bubble'>{ans}</div>", unsafe_allow_html=True)
        if summary_mode:
            summary = ask_claude("Summarize this answer in 3 short bullet points:", ans)
            st.markdown("### 🧾 Summary Mode")
            st.markdown(f"<div class='chat-bubble'>{summary}</div>", unsafe_allow_html=True)

st.markdown("<hr><p style='text-align:center;color:#999;'>🤖 Nestor’s Bot</p>", unsafe_allow_html=True)

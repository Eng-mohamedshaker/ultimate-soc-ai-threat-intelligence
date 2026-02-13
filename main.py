# =========================================================
# 🚨 Ultimate Enterprise SOC 6.6 – Granite Auto-Load
# Author: Mohamed Shaker
# Live MCP | Granite AI | Timeline | Heatmap | Auto-Block
# =========================================================

import os, json, hashlib, time, random, gc, threading
from collections import Counter, deque
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import psutil
import torch

# ==========================
# Transformers Import
# ==========================
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()  # منع التحذيرات الكثيرة

# ==========================
# Auto Refresh (LIVE)
# ==========================
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=1000, key="soc_live_refresh")

# ==========================
# Paths
# ==========================
BASE_DIR = os.path.dirname(__file__)
MODEL_CACHE = os.path.join(BASE_DIR, "model_cache")
AI_CACHE = os.path.join(BASE_DIR, "ai_cache")

os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(AI_CACHE, exist_ok=True)

# ==========================
# Streamlit Config
# ==========================
st.set_page_config("Ultimate SOC 6.6", layout="wide")
st.title("🚨 Ultimate Enterprise SOC 6.6 – LIVE")
st.caption("🔴 MCP Live • 🧠 Granite AI • 📈 Timeline • 🌡️ Heatmap • 🚫 Auto‑Block")

# ==========================
# Session State
# ==========================
if "events" not in st.session_state:
    st.session_state.events = deque(maxlen=2000)
if "ai_text" not in st.session_state:
    st.session_state.ai_text = ""

EVENTS = st.session_state.events

# ==========================
# Threat Knowledge
# ==========================
PORT_KB = {
    22: ("SSH", "High", "T1021"),
    80: ("HTTP", "Medium", "T1190"),
    443: ("HTTPS", "Low", "Normal"),
    3389: ("RDP", "Critical", "T1021"),
    3306: ("MySQL", "High", "Data Access"),
}
COUNTRIES = ["USA", "DEU", "EGY", "RUS", "CHN", "GBR", "FRA", "SAU"]

def generate_event():
    port = random.choice(list(PORT_KB))
    service, risk, mitre = PORT_KB[port]
    return {
        "time": datetime.utcnow(),
        "ip": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
        "country": random.choice(COUNTRIES),
        "port": port,
        "service": service,
        "risk": risk,
        "mitre": mitre
    }

# ==========================
# MCP LIVE ENGINE
# ==========================
def mcp_engine():
    while True:
        EVENTS.append(generate_event())
        if psutil.virtual_memory().percent > 80 and len(EVENTS) > 1000:
            EVENTS.popleft()
        time.sleep(0.5)

if "mcp_thread" not in st.session_state:
    t = threading.Thread(target=mcp_engine, daemon=True)
    t.start()
    st.session_state.mcp_thread = t

# ==========================
# Granite AI Engine
# ==========================
tokenizer = None
model = None
model_loaded = False

def load_ai_granite():
    global tokenizer, model, model_loaded

    if model_loaded:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ibm-granite/granite-4.0-micro"

    # تحقق من وجود الموديل في model_cache
    model_folder = os.path.join(MODEL_CACHE, "granite-4.0-micro")
    if os.path.exists(model_folder):
        st.sidebar.info("✅ Loading Granite AI from local cache...")
        tokenizer = AutoTokenizer.from_pretrained(model_folder)
        model = AutoModelForCausalLM.from_pretrained(model_folder).to(device)
    else:
        st.sidebar.info("⬇️ Downloading Granite AI to model_cache...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE).to(device)
        st.sidebar.success("✅ Granite AI downloaded and cached.")

    model_loaded = True

def ai_analyze(events):
    if not events:
        return "No live events yet."
    load_ai_granite()

    summary = Counter([e["risk"] for e in events])
    prompt = f"""
SOC LIVE ANALYSIS
Events: {len(events)}
Risks: {dict(summary)}
Top Ports: {Counter([e['port'] for e in events]).most_common(5)}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return text

# ==========================
# Sidebar System Info & Cache
# ==========================
mem = psutil.virtual_memory()
st.sidebar.markdown("### 🖥️ System")
st.sidebar.write(f"RAM: {mem.percent}%")
st.sidebar.write(f"CPU: {psutil.cpu_percent()}%")

if st.sidebar.button("🗑️ Clear AI Cache"):
    for f in os.listdir(AI_CACHE):
        os.remove(os.path.join(AI_CACHE, f))
    st.sidebar.success("✅ AI Cache cleared")
    gc.collect()

# ==========================
# DataFrame
# ==========================
df = pd.DataFrame(EVENTS)

# ==========================
# Metrics
# ==========================
if not df.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Events", len(df))
    c2.metric("Critical", len(df[df["risk"] == "Critical"]))
    c3.metric("Severity", min(100, len(df[df["risk"].isin(["High","Critical"])]) * 5))

# ==========================
# TABS (LIVE)
# ==========================
tabs = st.tabs([
    "📊 Live Events",
    "🧠 AI Analysis",
    "📈 Timeline",
    "🌡️ Risk Heatmap",
    "🚫 Auto Block & MITRE"
])

# --------------------------
# Live Events
# --------------------------
with tabs[0]:
    st.dataframe(df.tail(50), use_container_width=True)

# --------------------------
# AI Analysis (LIVE)
# --------------------------
with tabs[1]:
    if not df.empty:
        st.session_state.ai_text = ai_analyze(EVENTS)
        st.markdown(
            f"<pre style='background:#0f172a;color:#22ffcc;padding:15px;border-radius:12px'>{st.session_state.ai_text}</pre>",
            unsafe_allow_html=True
        )

# --------------------------
# Timeline
# --------------------------
with tabs[2]:
    if not df.empty:
        fig = px.histogram(df, x="time", color="risk", title="📈 Live Threat Timeline")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Heatmap
# --------------------------
with tabs[3]:
    if not df.empty:
        heat = df.groupby("country").size().reset_index(name="Events")
        fig = px.bar(heat, x="country", y="Events", color="Events", title="🌡️ Risk Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Auto Block
# --------------------------
with tabs[4]:
    bad = df[df["risk"].isin(["High", "Critical"])]["ip"].unique()
    if len(bad):
        st.error("🚫 Auto‑Block Candidates")
        st.write(list(bad))
    else:
        st.success("✅ No threats require blocking")

    st.subheader("🎯 MITRE ATT&CK")
    st.dataframe(df[["port","mitre","risk"]].drop_duplicates())

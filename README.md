Ultimate Enterprise SOC 
AI-Powered Real-Time Threat Intelligence Platform

📌 Overview
Ultimate SOC is an AI-powered real-time Security Operations Center (SOC) simulation platform designed to demonstrate enterprise-grade threat monitoring, automated risk scoring, intelligent mitigation logic, and AI-assisted threat analysis using IBM Granite 4.0.

The platform simulates live network telemetry and applies LLM-based reasoning to generate contextual threat intelligence insights.

🧠 What This Project Does
The system performs the following operations in real time:

1️⃣ Continuously generates simulated network security events
2️⃣ Classifies events by service, country, risk level, and MITRE ATT&CK technique
3️⃣ Maintains a memory-optimized event queue
4️⃣ Calculates dynamic severity scoring
5️⃣ Visualizes threat activity (timeline + heatmap)
6️⃣ Identifies high-risk IPs for auto-blocking
7️⃣ Sends aggregated telemetry to Granite 4.0 LLM
8️⃣ Generates AI-powered SOC analysis and mitigation recommendations

⚙️ How It Works (Architecture Flow)
MCP Engine (Threaded)
        ↓
Event Queue (Deque)
        ↓
DataFrame Processing
        ↓
Risk Scoring + MITRE Mapping
        ↓
Granite AI Analysis (LLM)
        ↓
Visualization Dashboard (Streamlit + Plotly)
        ↓
Auto-Block Logic
🔥 Core Components
🔴 MCP Live Engine
Thread-based event generator

Simulates IP, port, service, country

Auto memory trimming

🧠 Granite AI Engine
Uses ibm-granite/granite-4.0-micro

Local inference via HuggingFace Transformers

Structured SOC analysis prompt

Risk summarization & mitigation output

📊 Visualization Layer
Real-time event table

Risk timeline histogram

Country heatmap

Dynamic severity metrics

🚫 Auto-Block Engine
Detects repeated High/Critical threats

Generates block candidates

MITRE ATT&CK technique mapping

🛠 Technical Stack
Python 3.10+

Streamlit

Plotly

Pandas

PyTorch

Transformers (HuggingFace)

Psutil

Threading

Deque (memory optimized queue)

📦 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/Eng-mohamedshaker/ultimate-soc-ai-threat-intelligence.git
cd ultimate-soc-ai-threat-intelligence
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
3️⃣ Install Required Libraries
pip install -r requirements.txt
Or manually:

pip install streamlit pandas plotly psutil torch transformers streamlit-autorefresh
4️⃣ Run the Application
streamlit run app.py
(Replace app.py with your actual file name if different.)

🧠 AI Model Details
Model used:

ibm-granite/granite-4.0-micro
The model:

Runs locally

Uses GPU if available

Falls back to CPU automatically

Generates contextual SOC intelligence output

📈 Performance Optimization
Memory auto-trimming

GPU cache clearing

AI execution throttling

Threaded event engine

Controlled max token generation

🎯 Project Goals
Demonstrate AI-assisted SOC automation

Simulate enterprise SIEM behavior

Integrate LLM reasoning into cybersecurity workflows

Reduce manual analyst workload

Prototype next-gen AI-driven SOC systems

🚀 Future Enhancements
Azure AI integration

Microsoft Sentinel connector

Persistent database logging

WebSocket-based real-time backend

SOAR automation playbooks

Multi-tenant SaaS architecture

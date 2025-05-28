Customer Support Chatbot
A full-stack customer support chatbot built with React (Vite), Tailwind CSS, and FastAPI, leveraging Hugging Face’s Zero-Shot Classification, the Sentence-Transformers library for FAQ retrieval, and a simple rule-based layer for greetings and escalations.
🚀 Quick Start
1. **Clone the repo**
```bash
git clone https://github.com/aaryanb090/customer-support-agent.git
cd customer-support-agent
```

2. **Hugging Face API Token**
- Sign up at huggingface.co and go to **Settings → Access Tokens**.
- Create a **Fine-grained** token with at least:
  - **Make calls to Inference Providers**
  - **Read access to public models**
- Copy your token.

3. **Backend setup**
```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
echo HF_API_TOKEN=<your_token> > .env
uvicorn main:app --reload
```

4. **Frontend setup**
```bash
cd ../frontend
npm install
npm run dev
```
Your React app will open at http://localhost:5173.
🏗 Architecture & Flow
1. **User types a message** in the chat window and presses **Send**.
2. **Frontend** POSTs to `/agent` with JSON:
```json
{
  "user_id": "u1",
  "message": "...",
  "top_k": 3
}
```
3. **FastAPI** receives the request and processes:
- **Greeting Detector**
- **Short-utterance guard**
- **Urgent Escalation**
- **Rule-based Intent**
- **Zero-Shot Classification**
- **Small-talk fallback**
- **Other-intent fallback**
- **Technical Support branch**
- **Final support fallback**
🔧 Configuration & Thresholds
- **Zero-Shot labels**: "technical support request", "product feature suggestion", "sales inquiry", "small talk / general inquiry"
- **Regex thresholds**: Support regex → force Technical Support; Sales regex → force Sales Lead; Feature regex → force Feature Request.
- **FAISS score cutoff**: ≥ 0.20
- **Sentiment escalation**: NEGATIVE & score > 0.70
- **Short-utterance**: len(text) < 3
- **Urgent keywords**: immediate escalation keywords like "refund", "error", "bug", etc.
📂 File Structure
```
.
├── backend
│   ├── main.py
│   ├── .env
│   ├── kb/
│   ├── feature_requests.csv
│   ├── missing_docs.csv
│   └── requirements.txt
└── frontend
    ├── src/
    │   ├── components/
    │   └── index.css
    ├── index.html
    ├── package.json
    └── tailwind.config.cjs
```

# MAI-DxO Streamlit App

A Streamlit interface for the MAI Diagnostic Orchestrator - an AI-powered medical diagnosis system with 8 specialized physician agents.

## Quick Setup

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd streamlit-mai-dxo
```

### 2. Install Dependencies with UV
```bash
uv sync
```

### 3. Configure API Keys
Create a `.env` file:
```bash
OPENAI_API_KEY="your_openai_key_here"
GEMINI_API_KEY="your_gemini_key_here"  # Optional
ANTHROPIC_API_KEY="your_anthropic_key_here"  # Optional
```

### 4. Run Locally
```bash
uv run streamlit run streamlit_app.py
```

### 5. Deploy to Streamlit Cloud

1. Push to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub repo

4. Add your API keys in the Streamlit Cloud secrets:
   - Go to your app settings
   - Add secrets in TOML format:
```toml
OPENAI_API_KEY = "your_key_here"
```

## Features

- 8 AI Physician Agents working together
- Multiple AI models (GPT-4o, Gemini, Claude)
- 5 diagnostic modes (No Budget, Budgeted, Question Only, Instant, Ensemble)
- Real-time cost tracking and accuracy scoring
- Export results as JSON

## Usage

1. Select AI model and diagnostic mode in sidebar
2. Enter patient case information in tabs
3. Click "Run Diagnosis" 
4. View results with reasoning and metrics
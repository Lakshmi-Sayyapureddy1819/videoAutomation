# Vid AI Automation

YouTube search → download → AI processing (CLIP + MoviePy) → final videos.

## Structure

```
vid-ai-automation/
├── .streamlit/               # Streamlit configuration
│   └── config.toml           # UI styling (theme, port, etc.)
├── data/                     # Persistent storage
│   ├── raw_videos/           # Temporary YT downloads
│   └── output/               # Final generated videos
├── models/                   # Local weights (CLIP)
├── src/
│   ├── app.py                # Main Streamlit frontend
│   ├── processor.py          # AI logic (CLIP + MoviePy)
│   └── scout.py              # YouTube search/download
├── .env                      # API keys (local only, DO NOT UPLOAD)
├── .env.example              # Template for keys
├── .gitignore                # Excludes /data, .env, /models
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Setup

1. **Clone and enter the project**
   ```bash
   cd videoAutomation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy `.env.example` to `.env`
   - Add your API keys (e.g. YouTube) in `.env`

5. **Run the app**
   ```bash
   streamlit run src/app.py
   ```

## Notes

- Keep `.env` and `data/` out of version control (see `.gitignore`).
- Put CLIP or other model weights in `models/` (gitignored).

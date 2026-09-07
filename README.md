# MarketMind AI — Simple Streamlit Version

This is a compact implementation of the MarketMind AI coursework application.

## Folder structure

```text
MarketMind/
├── app.py
├── requirements.txt
├── README.md
├── corpus/
│   └── demo.json
└── runs/
```

## What happens when the app starts?

### 1. OpenAI connection page

If no API key is configured, the first page asks the user to connect an OpenAI API key.

There is also a button to open the OpenAI API key page:

https://platform.openai.com/api-keys

The app makes a small real API request to verify the connection.

### 2. Research page

After successful connection, the user enters a natural-language business research question.

MarketMind then performs:

```text
User question
      ↓
Research planning
      ↓
Local corpus search
      ↓
Document retrieval
      ↓
Evidence collection
      ↓
Evidence-grounded synthesis
      ↓
Basic QC
      ↓
Human approval
      ↓
JSON run saved
```

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## API key

You can enter an API key on the first screen.

For Streamlit Cloud, the recommended method is:

**App → Settings → Secrets**

Add:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-5.6-luna"
```

Do not upload a `.env` file containing your real key to GitHub.

## Persistence

No database is used.

Research runs are saved as JSON files in:

```text
runs/
```

This keeps the application simple while providing serializable research state.

## Research corpus

`corpus/demo.json` is a controlled illustrative corpus.

The application does not present the synthetic corpus as live web research.

## Assignment alignment

The compact version includes the main concepts required by the assignment:

- OpenAI API integration
- request planning
- manual research workflow
- six required tool concepts
- evidence records
- provenance/source references
- evidence-grounded synthesis
- QC
- bounded workflow
- human approval
- serializable state persistence
- prompt-injection handling
- local reproducible corpus

For a production system, the compact file structure would normally be split into separate modules.

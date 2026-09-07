# MarketMind AI

A compact Streamlit implementation of the **MarketMind AI — Autonomous Business Research Agent** assignment.

## Project structure

```text
MarketMind/
├── app.py
├── requirements.txt
├── README.md
├── corpus/
│   └── demo.json
└── runs/
```

## What is included

- Natural-language research request
- Research planning
- Manually implemented bounded research loop
- Six required tools:
  - `search_information`
  - `retrieve_document`
  - `calculate_metric`
  - `compare_companies`
  - `save_research`
  - `generate_report`
- Tool argument validation
- Structured Pydantic schemas
- Evidence records and source references
- Prompt-injection handling for retrieved content
- Competitor comparison
- Quality-control checks
- Bounded QC repair
- Final report validation
- Human approval gate
- JSON state persistence in `runs/`
- Demo/local corpus mode
- Optional OpenAI API mode

## 1. Install

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
streamlit run app.py
```

Demo mode works without an API key.

## 3. OpenAI mode

Set:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.6-luna
```

You can put these in environment variables or a local `.env` file.

**Never upload `.env` or your API key to GitHub.**

For Streamlit Cloud, put the key in the application's Secrets settings.

## Persistence

The assignment requires research state to be serializable/persistable so a run can be resumed/audited. This project uses JSON files in `runs/` instead of a database.

## Corpus

`corpus/demo.json` is a controlled illustrative corpus. It should not be presented as live market truth.

## Important assignment design choices

The application does not use a prebuilt agent executor for the core research loop. Tool requests are validated and dispatched by application code.

Retrieved content is treated as untrusted data. Prompt-injection text in the corpus is flagged and is never treated as an instruction.

The report remains pending until a human makes an approval decision.

## GitHub

The recommended repository is:

```text
MarketMind/
├── app.py
├── requirements.txt
├── README.md
├── corpus/
│   └── demo.json
└── runs/
```

Do not upload secrets.

# Deployment: live demo on Streamlit Cloud + Qdrant Cloud

A 10-minute walkthrough to ship a public demo your colleagues can hit.

## Prerequisites

- GitHub account
- An Anthropic API key
- A Qdrant Cloud account (free tier — 1GB): https://cloud.qdrant.io

## 1. Provision Qdrant Cloud

1. Sign up → "Create cluster" → Free tier, any region close to you.
2. Wait ~1 minute for provisioning.
3. Copy the **cluster URL** (looks like `https://xxxxxxxx.us-east.aws.cloud.qdrant.io:6333`) and an **API key**.

## 2. Ingest into Qdrant Cloud (one time, from your laptop)

```bash
cp .env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY=sk-ant-...
#   QDRANT_URL=<your Qdrant Cloud URL>
#   QDRANT_API_KEY=<your Qdrant Cloud key>

python scripts/download_filings.py
python scripts/ingest.py
```

Verify:

```bash
curl -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections/financial_filings"
```

Should report several thousand points.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Initial production RAG"
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main
```

Make sure `.env` and `.streamlit/secrets.toml` are gitignored (they already are).

## 4. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io → "New app".
2. Pick your repo + branch `main` + main file `streamlit_app.py`.
3. Click **Advanced settings** → **Secrets** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
QDRANT_URL = "https://xxxxxxxx.us-east.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "<your Qdrant Cloud key>"
QDRANT_COLLECTION = "financial_filings"
GENERATION_MODEL = "claude-sonnet-4-6"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"
```

4. Deploy. First load is slow (~2 min) while FastEmbed downloads the BGE and reranker ONNX models. Subsequent loads are cached.

## 5. Share the URL

Streamlit gives you a URL like `https://your-app.streamlit.app`. Send it around. You're done.

## Troubleshooting

| Symptom | Fix |
|---|---|
| App crashes on boot with "OOM" | Lower the reranker top-K, or remove the reranker by setting `EMBEDDING_MODEL` to skip it. Streamlit Community tier is 1GB. |
| First query is very slow | Cold start. FastEmbed loads ONNX models on first call. Subsequent queries are fast. |
| "ANTHROPIC_API_KEY is empty" warning in logs | Double-check the Streamlit secrets — the secret key names must match exactly. |
| Qdrant connection refused | Confirm `QDRANT_URL` includes the port (`:6333`) and starts with `https://`. |

## Cost ballpark

For light demo traffic (a few colleagues, a few dozen queries / day):

- Streamlit Community Cloud: **$0**
- Qdrant Cloud free tier: **$0**
- Anthropic API: cents per day — most prompt tokens are cache reads, so cost is dominated by output tokens (~$15 / 1M output for Sonnet 4.6).

Total: typically **< $1/day**.

## When to upgrade

- More than ~10k chunks: bump Qdrant Cloud out of free tier
- More than 1GB working set: move off Streamlit Community to a paid tier or self-host on Fly.io / Railway
- Strict latency SLOs: pre-warm the FastEmbed models on a long-running container instead of Streamlit's per-instance lifecycle

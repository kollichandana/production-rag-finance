.PHONY: install qdrant ingest serve ui eval benchmark test lint clean docker-up docker-down

install:
	pip install -r requirements.txt && pip install -e .

qdrant:
	docker run -d --name rag-qdrant -p 6333:6333 -p 6334:6334 -v $$(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.4

download:
	python scripts/download_filings.py

ingest:
	python scripts/ingest.py --input data/raw --collection $${QDRANT_COLLECTION:-financial_filings}

serve:
	uvicorn rag.api.main:app --reload --port 8000

ui:
	streamlit run streamlit_app.py

eval:
	python scripts/run_eval.py

benchmark:
	python scripts/run_eval.py --compare-naive

test:
	pytest -v

lint:
	ruff check src tests scripts

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ .ruff_cache .pytest_cache *.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} +

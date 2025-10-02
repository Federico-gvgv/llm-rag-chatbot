PY=python


setup:
$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt


ingest:
. .venv/bin/activate && $(PY) ingest.py --config config/config.yaml


eval:
. .venv/bin/activate && $(PY) eval.py --config config/config.yaml


run:
. .venv/bin/activate && streamlit run app.py --server.runOnSave=true


# optional docker
build-docker:
docker build -t llm-chatbot -f docker/Dockerfile .
run-docker:
docker run -it --rm -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 \
-v $(PWD):/app llm-chatbot
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e . \
    && mkdir -p data chroma_db logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "clinical_rag_eval.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/ artifacts/
COPY configs/ configs/

ENV MODEL_ARTIFACT_PATH=artifacts/model.joblib
ENV MODEL_METADATA_PATH=artifacts/model_metadata.json
ENV CREDIT_SCORING_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]

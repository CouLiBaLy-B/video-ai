FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/apps/api/src

COPY pyproject.toml README.md ./
COPY apps/api ./apps/api
RUN pip install --no-cache-dir -e '.[dev,worker,db,storage]'

EXPOSE 8000
CMD ["uvicorn", "video_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

# Create a non-root user (Hugging Face requirements)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files
COPY --chown=user . .

# HF Spaces uses port 7860
EXPOSE 7860

# Run the FastAPI server (using the new spec-compliant path)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

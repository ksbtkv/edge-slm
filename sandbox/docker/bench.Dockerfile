# Thin image that runs the Stage 0 harness against the ollama service.
# Kept CPU-only and tiny; on a CUDA host the compose override grants it the
# nvidia 'utility' capability so nvidia-smi works for VRAM reads.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /work

COPY harness/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY harness/ ./harness/

ENTRYPOINT ["python", "harness/stage0_bench.py"]
CMD ["--label", "clean"]

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The port is Hugging Face's, not ours: a Docker Space routes to the port its
# README's app_port names, and with none it is the one Hugging Face states as
# the default exposed port (https://huggingface.co/docs/hub/spaces-sdks-docker).
EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:server"]

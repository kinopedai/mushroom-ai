FROM python:3.11-slim

WORKDIR /workspace

# 依存が増えたら以下のコメントアウトを解除
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

CMD ["python", "train.py"]

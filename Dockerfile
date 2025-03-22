FROM python:3.9-slim
WORKDIR /cfd
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run.py"]
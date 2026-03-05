FROM python:3.10-slim

WORKDIR /app

# Python'un src klasörünü bulabilmesi için yolu ekleyelim
ENV PYTHONPATH=/app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 8501

CMD ["bash", "start.sh"]

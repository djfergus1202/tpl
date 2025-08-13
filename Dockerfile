
FROM python:3.10-slim

WORKDIR /app

# System deps for scientific stack (lean)
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Editable install
RUN pip install -e .

EXPOSE 8501
CMD ["python", "-m", "pharma_lab_suite"]

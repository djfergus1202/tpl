
# Pharma Lab Suite

Functional, GitHub-deployable research suite for data analysis, molecular modeling, pharmacological topological maps, and paper generation.

## Quick Start (Local)

```bash
cd pharma_lab_suite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pharma_lab_suite
```

The app will launch Streamlit at http://localhost:8501.

## Docker

```bash
docker build -t pharma-lab-suite:latest .
docker run -p 8501:8501 pharma-lab-suite:latest
```

## Kubernetes

1. Push your image to a registry (replace image name in `kubernetes-deployment.yaml`).
2. Deploy:

```bash
kubectl apply -f kubernetes-deployment.yaml
```

## GitHub

- Push this folder to a GitHub repo.
- Optionally add a GitHub Actions workflow to build/push the Docker image.

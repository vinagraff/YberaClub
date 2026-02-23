## Deploy no GCP (Cloud Run Free Tier)

### 1) Pré-requisitos
- Conta GCP com faturamento habilitado (necessário mesmo no free tier).
- `gcloud` instalado e autenticado.
- Projeto GCP criado.

### 2) Configurar projeto e APIs
```bash
gcloud auth login
gcloud config set project SEU_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 3) Deploy direto do código (usando Dockerfile)
Na raiz do projeto:
```bash
gcloud run deploy ybera-club \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 20
```

### 4) Custos (free tier)
- Cloud Run free tier é mensal, por uso.
- Para reduzir custo: `--min-instances 0` e `--max-instances 1`.
- Região `us-central1` costuma ser boa escolha para free tier.

### 5) Atualizar versão
Após mudanças no código:
```bash
gcloud run deploy ybera-club --source . --region us-central1 --allow-unauthenticated
```

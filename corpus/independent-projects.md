# Independent AI/ML Portfolio Project (Self-Directed, Outside of Core Role)

This is Anmol's self-directed transition project toward AI/ML engineering — built independently, in parallel with his WPP Media role, not as part of any employer's work.

## Production infrastructure
- Designed and deployed a production infrastructure chassis: Docker, Terraform (infrastructure as code), Kubernetes (AWS EKS), GitHub Actions CI/CD, and Langfuse for observability/tracing.
- This chassis is built to host a portfolio of AI agents — each new agent is deployed through the same reusable pipeline rather than building infrastructure from scratch each time.
- Practiced real production failure modes deliberately: Terraform state drift and reconciliation, forced CI/CD failures and fixes, debugging real issues (e.g. an outdated dependency with a security CVE, and an ARM64/AMD64 container architecture mismatch between a local Mac build and AWS EKS nodes).

## RAG & agent work
- Building Retrieval-Augmented Generation (RAG) systems for marketing/media analytics use cases (e.g., cross-campaign performance Q&A), including GraphRAG (Neo4j-based knowledge graphs) and RAPTOR-based synthesis, applying techniques from a completed Advanced-RAG/GraphRAG learning project benchmarked on Unilever media-mix data.
- Orchestrating multi-agent systems using LangGraph and CrewAI, and MCP-based tool integration.

## Fine-tuning & MLOps
- Fine-tuning open-source LLMs (Unsloth, QLoRA) for domain-specific tasks, with experiment tracking in MLflow.
- Applying end-to-end MLOps practices — containerization, CI/CD, monitoring — across the full training-to-deployment lifecycle.

## Why this project
Anmol built this portfolio specifically to develop hands-on, production-grade AI/ML engineering skills that go beyond his analytics background — with the explicit goal of being able to speak in depth, from direct experience, about production AI systems in interviews for MLE/MLOps roles.
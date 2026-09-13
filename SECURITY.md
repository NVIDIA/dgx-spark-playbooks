# Security Policy: DGX Spark Playbooks

## Reporting a Vulnerability

If you discover a potential security vulnerability in DGX Spark Playbooks, please **do not open a public issue or pull request**. Public reports can expose details before maintainers have had time to assess and address the issue.

Report potential vulnerabilities through one of these private channels:

- **NVIDIA Vulnerability Disclosure Program** (preferred): https://www.nvidia.com/en-us/security/
- **Email**: [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - We encourage use of the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key) for secure email communication.
- **GitHub Private Vulnerability Reporting**: Use this repository's **Security** tab > **Report a vulnerability**.

Please include the following information where possible:

- Product/project name, playbook name, affected file path, and version or commit
- Type of vulnerability, such as unsafe deployment defaults, credential exposure, command injection, insecure network exposure, or dependency issue
- Step-by-step reproduction instructions
- Proof-of-concept details if available, avoiding public disclosure
- Expected and actual behavior
- Impact assessment, including whether the issue affects local-only demos, network-exposed services, credentials, user-uploaded data, or DGX Spark host configuration

Detailed reports help NVIDIA evaluate and address issues faster. NVIDIA's PSIRT team will acknowledge receipt, validate severity, coordinate fixes, and publish security bulletins as appropriate.

## Security Architecture & Context

DGX Spark Playbooks is a collection of step-by-step documentation, scripts, notebooks, Docker configurations, and sample applications for setting up AI/ML workloads on NVIDIA DGX Spark devices. The repository includes local deployment examples for model serving, RAG workflows, GPU-accelerated demos, multi-node networking setup, fine-tuning, web UIs, and supporting utility scripts.

This software operates primarily at the **documentation, sample application, CLI/script, and local service deployment** level. Its primary security responsibility is to provide safe setup guidance and sample code that does not unnecessarily expose DGX Spark hosts, model-serving endpoints, user-uploaded documents, chat history, credentials, or cluster administration workflows.

**Repository Exposure Classification:** Public.
Basis: GitHub reports this NVIDIA repository as publicly visible; the document is written to public-safe detail.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: repository contains externally distributed DGX Spark playbooks and runnable sample applications, including local web services, model-serving containers, credential-handling examples, file ingestion, chat history storage, and cluster setup automation.

The repository is not a single production service. It contains multiple playbooks with different trust boundaries:

- The root `README.md` advertises the repository as a collection of DGX Spark playbooks for installing frameworks, running inference, setting up development environments, and managing devices.
- Several playbooks are documentation-only or notebook-oriented and rely on the user to run commands in a local DGX Spark environment.
- `nvidia/multi-agent-chatbot/assets` contains a full-stack local chatbot application with a FastAPI backend, Next.js frontend, PostgreSQL conversation storage, Milvus vector storage, MinIO object storage, local model-serving containers, WebSocket chat, file upload, image upload, and MCP tool execution.
- `nvidia/multi-sparks-through-switch/assets/spark_cluster_setup` contains CLI automation that reads node configuration, connects over SSH, runs privileged setup commands, copies keys, configures networking, and runs cluster validation.
- Other playbooks include Dockerfiles, docker-compose files, shell scripts, notebooks, and web/API examples for local model serving, visualization, recommendation, healthcare, graph, and fine-tuning workflows.

Key security boundaries include:

- **User workstation/browser to local demo services:** Playbooks commonly expose local web UIs and APIs on localhost or container ports. These examples assume the user controls the host and network exposure.
- **Uploaded documents/images to RAG and vision pipelines:** The multi-agent chatbot accepts user-selected files and images, stores them locally, indexes text into Milvus, stores chat/image data in PostgreSQL, and passes content into local LLM/VLM services.
- **Frontend to backend APIs and WebSocket:** The multi-agent chatbot frontend connects to backend HTTP endpoints and `/ws/chat/{chat_id}` for streaming chat responses.
- **Backend to local model services:** The chatbot backend and MCP tools call local OpenAI-compatible model endpoints over container networking.
- **Automation scripts to DGX Spark nodes:** The multi-node setup scripts use SSH credentials, write SSH configuration, install packages, configure networking, and execute commands on remote nodes.
- **Documentation examples to user environments:** Many playbooks instruct users to export tokens, run containers with environment variables, download gated models, or grant Docker access; these commands execute with the user's local privileges.

### Threat Model

The following scenarios represent the primary security concerns for this repository and its runnable examples:

1. **Unauthenticated local chatbot APIs expose uploaded content and chat history:** `nvidia/multi-agent-chatbot/assets/backend/main.py` defines FastAPI routes for file ingestion, image upload, chat listing, chat deletion, collection deletion, model selection, and WebSocket chat without application-level authentication. If the backend is bound beyond a trusted local interface or reachable by another local user, an attacker could read or manipulate conversations, uploaded images, selected sources, and indexed collections.

2. **Untrusted document ingestion can write and process attacker-controlled files:** `nvidia/multi-agent-chatbot/assets/backend/main.py` reads uploaded files from `/ingest`, and `nvidia/multi-agent-chatbot/assets/backend/utils.py` writes each provided filename under an uploads directory before parsing and indexing it. Malicious filenames or crafted documents could affect local filesystem contents, parser behavior, indexing reliability, or resource consumption.

3. **Model and MCP tool orchestration can amplify prompt-injection effects:** `nvidia/multi-agent-chatbot/assets/backend/agent.py` lets the supervisor model select MCP tools, including RAG, image understanding, weather, and code generation tools registered through `nvidia/multi-agent-chatbot/assets/backend/client.py`. Content from user prompts, uploaded documents, and uploaded images can influence tool calls and generated output, so prompt-injection or data-poisoning content may cause misleading answers or unintended tool use within the local demo boundary.

4. **Cluster setup automation handles sensitive SSH credentials and runs privileged commands:** `nvidia/multi-sparks-through-switch/assets/spark_cluster_setup/spark_cluster_setup.py` reads node configuration containing SSH connection details, connects with Paramiko, modifies SSH state, installs packages, configures networking, and executes privileged commands. Incorrectly protected config files or maliciously modified playbook assets could expose credentials or cause unintended changes on DGX Spark nodes.

5. **Default demo credentials and exposed container ports can become unsafe outside a local lab:** `nvidia/multi-agent-chatbot/assets/docker-compose.yml` configures PostgreSQL and MinIO with demo credentials and maps backend, frontend, database, and Milvus ports to the host. These defaults are appropriate only for isolated local demonstrations and should not be reused for shared, production, or internet-accessible deployments.

6. **Dependency and container supply chain drift affects runnable examples:** The repository includes Python, Node.js, Docker, and notebook-based examples with dependencies such as FastAPI, LangChain, MCP adapters, unstructured document parsing, Next.js canary builds, model-serving containers, database services, and GPU libraries. Users who build or run examples inherit security posture from upstream packages, container images, and downloaded models.

7. **Logging may capture sensitive user-provided data or operational details:** `nvidia/multi-agent-chatbot/assets/backend/logger.py`, `main.py`, `agent.py`, `vector_store.py`, and MCP server code log request metadata, filenames, selected sources, tool calls, errors, and portions of model interactions. In shared environments, logs could disclose uploaded document names, prompts, model behavior, or operational context.

### Critical Security Assumptions

- **Local-only deployment is trusted:** The runnable web applications and container stacks are assumed to be used on a trusted local DGX Spark host or behind access controls. The sample services are not designed as internet-facing production services without additional hardening.
- **Users protect credentials and tokens:** Playbook examples that use Hugging Face, NGC, API, database, MinIO, or SSH credentials assume users provide real secrets through secure local mechanisms and do not commit populated configuration files or shell history containing secrets.
- **Uploaded files are intentionally provided by a trusted operator:** File ingestion examples assume the user controls the files being uploaded. The sample code does not establish a strong sandbox for arbitrary hostile documents.
- **Container networks and host port mappings are operator-controlled:** Docker Compose examples assume users understand which ports are exposed to the host and restrict network access when running demos on shared or remote systems.
- **Model outputs and tool calls require user judgment:** Generated code, RAG answers, and model/tool responses are not treated as authoritative security decisions. Users are expected to review generated commands, code, and configuration before execution.
- **The host OS, Docker daemon, GPU driver stack, and filesystem permissions enforce isolation:** The playbooks rely on the underlying operating system, container runtime, and DGX Spark platform controls for process, file, device, and network isolation.
- **Cluster automation runs only against intended nodes:** Multi-node setup scripts assume configuration files identify trusted DGX Spark systems and that SSH credentials are protected with filesystem permissions appropriate for administrative material.
- **Third-party packages, models, and images are obtained from trusted sources:** The playbooks assume users validate upstream packages, model licenses, image provenance, and checksums where applicable before running downloaded artifacts.

## Supported Versions

Security fixes and guidance apply to the current `main` branch unless a release branch or tag is explicitly identified in a vulnerability report or NVIDIA security bulletin.

## Deployment Guidance

These playbooks are examples and should be hardened before use in shared, production, customer-facing, or internet-accessible environments:

- Bind demo services to loopback or a trusted interface unless remote access is required.
- Replace demo database, object-store, and service credentials before any shared deployment.
- Do not expose model-serving, database, vector-store, object-store, or administrative ports directly to untrusted networks.
- Use TLS, authentication, authorization, and logging controls at an application gateway or service mesh when adapting examples beyond local use.
- Review uploaded-file handling, filename normalization, file size limits, parser sandboxing, and retention policies before accepting files from untrusted users.
- Protect SSH configuration and cluster setup JSON files with restrictive permissions.
- Review shell scripts, Dockerfiles, notebooks, and generated code before execution.

## Dependency Security

This repository includes examples across Python, TypeScript/JavaScript, shell, Docker, YAML, and notebooks. Security-sensitive dependency areas include FastAPI and Uvicorn web serving, LangChain/LangGraph agent orchestration, MCP adapters, unstructured/PDF parsing, asyncpg/PostgreSQL storage, Milvus vector search, MinIO object storage, Next.js/React frontends, model-serving containers, and GPU/AI frameworks.

When running a playbook, users should install dependencies in isolated environments, rebuild containers from trusted bases, monitor upstream security advisories, and refresh lockfiles or pinned images as part of normal maintenance.


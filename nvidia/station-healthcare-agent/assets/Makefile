.PHONY: up down logs status setup test test-full pull clean teardown help prereq ngc-login

COMPOSE := docker compose
SHELL := /bin/bash

# Source .env so port and GPU overrides propagate to recipes that don't run
# via docker compose (e.g. setup_sandbox.sh's openshell provider create).
ifneq (,$(wildcard ./.env))
include .env
export
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Prerequisites & auth ────────────────────────────────────────

prereq: ## Validate Docker, Node.js v22, OpenShell, disk, GPU, port 11434, NGC auth
	@echo "=== Prerequisite checks ==="
	@command -v docker >/dev/null 2>&1 \
		&& echo "  Docker:           ✓ $$(docker info --format '{{.ServerVersion}}' 2>/dev/null)" \
		|| { echo "  Docker:           ✗ not installed"; exit 1; }
	@NODE_MAJ=$$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/'); \
		if [ -z "$$NODE_MAJ" ]; then echo "  Node.js:          ✗ not installed (need v22+)"; exit 1; \
		elif [ "$$NODE_MAJ" -lt 22 ]; then echo "  Node.js:          ✗ v$$NODE_MAJ found, need v22+. Upgrade: curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs"; exit 1; \
		else echo "  Node.js:          ✓ $$(node --version)"; fi
	@command -v openshell >/dev/null 2>&1 \
		&& echo "  OpenShell:        ✓ $$(openshell --version 2>/dev/null)" \
		|| { echo "  OpenShell:        ✗ not installed"; exit 1; }
	@FREE=$$(df -BG / | awk 'NR==2 {print $$4}' | tr -d 'G'); \
		if [ "$$FREE" -lt 200 ]; then echo "  Disk free /:      ⚠ $${FREE}G (need 200G+ for 86G model + images + working space)"; \
		else echo "  Disk free /:      ✓ $${FREE}G"; fi
	@nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 \
		&& echo "  GPU:              ✓ $$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',')" \
		|| { echo "  GPU:              ✗ nvidia-smi not available"; exit 1; }
	@if ss -tlnp 2>/dev/null | grep -q ':$${OLLAMA_PORT:-11434} '; then \
		echo "  Port $${OLLAMA_PORT:-11434}:        ⚠ already bound (host Ollama?). Stop with 'sudo systemctl stop ollama' or set OLLAMA_PORT in .env"; \
	else \
		echo "  Port $${OLLAMA_PORT:-11434}:        ✓ free"; \
	fi
	@if grep -q '"nvcr.io"' ~/.docker/config.json 2>/dev/null; then \
		echo "  NGC docker login: ✓ (run 'make ngc-login' to refresh)"; \
	else \
		echo "  NGC docker login: ✗ not authenticated. Run 'make ngc-login' (requires NGC_API_KEY in .env)"; \
	fi
	@echo ""
	@echo "Prereqs OK. Run 'make ngc-login' if needed, then 'make up'."

ngc-login: ## Authenticate Docker against nvcr.io using NGC_API_KEY from .env
	@if [ -z "$(NGC_API_KEY)" ] || [ "$(NGC_API_KEY)" = "nvapi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" ]; then \
		echo "ERROR: NGC_API_KEY not set in .env. Get one at https://ngc.nvidia.com/setup/api-key"; \
		exit 1; \
	fi
	@echo "$(NGC_API_KEY)" | docker login nvcr.io --username '$$oauthtoken' --password-stdin

# ── Infrastructure ──────────────────────────────────────────────

up: ## Start all services (Ollama + OpenFold3)
	$(COMPOSE) up -d ollama openfold3
	@echo ""
	@echo "Waiting for Ollama health..."
	@$(COMPOSE) up model-pull
	@echo ""
	@echo "Services started. Run 'make status' to check health."

down: ## Stop all services
	$(COMPOSE) down

logs: ## Tail all service logs
	$(COMPOSE) logs -f

status: ## Show service health (uses OLLAMA_PORT/OPENFOLD_PORT from .env)
	@echo "=== Services ==="
	@$(COMPOSE) ps
	@echo ""
	@echo "=== Health Checks ==="
	@OPORT=$$($(COMPOSE) port ollama 11434 2>/dev/null | sed 's/.*://') ; \
		: $${OPORT:=$${OLLAMA_PORT:-11434}} ; \
		curl -sf "http://localhost:$$OPORT/" > /dev/null 2>&1 \
			&& echo "  Ollama (port $$OPORT):     ✓ healthy" || echo "  Ollama (port $$OPORT):     ✗ down"
	@FPORT=$$($(COMPOSE) port openfold3 8000 2>/dev/null | sed 's/.*://') ; \
		: $${FPORT:=$${OPENFOLD_PORT:-8000}} ; \
		curl -sf "http://localhost:$$FPORT/v1/health/ready" > /dev/null 2>&1 \
			&& echo "  OpenFold3 (port $$FPORT):  ✓ healthy" || echo "  OpenFold3 (port $$FPORT):  ✗ down (may still be loading)"
	@echo ""
	@echo "=== GPU ==="
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
		|| echo "  nvidia-smi not available"

pull: ## Pull/update model in Ollama
	$(COMPOSE) up model-pull

# ── OpenShell Sandbox ───────────────────────────────────────────

setup: ## Create sandbox and deploy all config (run after 'make up')
	@echo "=== Setting up OpenShell sandbox ==="
	@echo "Prerequisites: openshell CLI installed, gateway started"
	@echo ""
	bash scripts/setup_sandbox.sh

setup-local: ## Same as setup but bind gateway to 0.0.0.0 (no SSH tunnel needed)
	bash scripts/setup_sandbox.sh --local

restart: ## Restart OpenClaw gateway inside sandbox
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
		-o ConnectTimeout=10 \
		-o "ProxyCommand=openshell ssh-proxy --gateway-name openshell --name $${SANDBOX_NAME:-clinical-sandbox}" \
		"sandbox@openshell-$${SANDBOX_NAME:-clinical-sandbox}" \
		"bash /sandbox/clinical-intelligence/scripts/restart_sandbox.sh"

# ── Testing ─────────────────────────────────────────────────────

check: ## Verify sandbox config matches repo (skills, identity, helpers, memory)
	@export PATH="$$HOME/.local/bin:$$PATH"; \
	bash scripts/check_sandbox_config.sh $${SANDBOX_NAME:-clinical-sandbox}

test: ## Run test suite levels 1-3 (~3 min)
	bash scripts/test-all.sh --level 3 --verbose

test-full: ## Run all test levels including agent tests (~45 min)
	bash scripts/test-all.sh --level 5 --verbose

test-docker: ## Run tests inside a container
	$(COMPOSE) run --rm test --level 3 --verbose

# ── Cleanup ─────────────────────────────────────────────────────

teardown: ## Tear down sandbox, services, and gateway
	openshell sandbox delete $${SANDBOX_NAME:-clinical-sandbox} 2>/dev/null || true
	$(COMPOSE) down
	openshell gateway destroy 2>/dev/null || true
	@echo "Teardown complete."

clean: ## Remove test results, PDB caches, and dangling images
	rm -rf test-results/
	$(COMPOSE) down --volumes --remove-orphans
	docker image prune -f

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl jq openssh-client python3 python3-pip bash coreutils grep sed gawk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY scripts/ scripts/
COPY agents/ agents/
COPY skills/ skills/
COPY IDENTITY.md openclaw.json sandbox-policy.yaml ./

RUN chmod +x scripts/test-all.sh scripts/test-lib.sh

ENTRYPOINT ["bash", "scripts/test-all.sh"]
CMD ["--level", "3", "--verbose"]

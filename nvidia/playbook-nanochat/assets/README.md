# NanoChat playbook assets

Scripts and Dockerfiles for single-node training on supported hardware platforms.

| File | Use |
|------|-----|
| `Dockerfile` | Single-node image (PyTorch NGC + dependencies) |
| `setup.sh` / `launch.sh` | Single-node setup and launch |
| `speedrun_single.sh` | Single-node speedrun (default d24) |

Follow the playbook **Instructions** tab for single-node training. Upstream project: [karpathy/nanochat](https://github.com/karpathy/nanochat).

> Multi-node scripts (`Dockerfile.multinode`, `setup_multinode.sh`, `launch_multinode.sh`, `speedrun_multinode.sh`) remain in this directory for a future DGX Spark release and are not part of the published playbook yet.

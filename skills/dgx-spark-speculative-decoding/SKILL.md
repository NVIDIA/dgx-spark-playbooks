---
name: dgx-spark-speculative-decoding
description: Learn how to set up speculative decoding for fast inference on Spark — on NVIDIA DGX Spark. Use when setting up speculative-decoding on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/speculative-decoding/README.md -->
# Speculative Decoding

> Learn how to set up speculative decoding for fast inference on Spark

Speculative decoding speeds up text generation by using a **small, fast model** to draft several tokens ahead, then having the **larger model** quickly verify or adjust them.
This way, the big model doesn't need to predict every token step-by-step, reducing latency while keeping output quality.

**Outcome**: You'll explore speculative decoding using TensorRT-LLM on NVIDIA Spark using two approaches: EAGLE-3 and Draft-Target.
These examples demonstrate how to accelerate large language model inference while maintaining output quality.

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/speculative-decoding/README.md`
<!-- GENERATED:END -->

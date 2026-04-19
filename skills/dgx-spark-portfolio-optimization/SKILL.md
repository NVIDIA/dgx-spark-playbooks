---
name: dgx-spark-portfolio-optimization
description: GPU-Accelerated portfolio optimization using cuOpt and cuML — on NVIDIA DGX Spark. Use when setting up portfolio-optimization on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/portfolio-optimization/README.md -->
# Portfolio Optimization

> GPU-Accelerated portfolio optimization using cuOpt and cuML

This playbook demonstrates an end-to-end GPU-accelerated workflow using NVIDIA cuOpt and NVIDIA cuML to solve large-scale portfolio optimization problems, using the Mean-CVaR (Conditional Value-at-Risk) model, in near real-time. 

Portfolio Optimization (PO) involves solving high-dimensional, non-linear numerical optimization problems to balance risk and return. Modern portfolios often contain thousands of assets, making traditional CPU-based solvers too slow for advanced workflows. By moving the computational heavy lifting to the GPU, this solution dramatically reduces computation time.

**Outcome**: You will implement a pipeline that provides tools for performance evaluation, strategy backtesting, benchmarking, and visualization. The workflow includes:
- **GPU-Accelerated Optimization:** Leveraging NVIDIA cuOpt LP/MILP solvers 
- **Data-Driven Risk Modeling:** Implementing CVaR as a scenario-based risk measure that models tail risks without making assumptions about asset return distributions.
- **Scenario Generation:** Using GPU-accelerated Kernel Density Estimation (KDE) via NVIDIA cuML to model return distributions.
- **Real-World Constraint Management:** Implementing constraints including concentration limits, leverage constraints, turnover limits, and cardinality constraints.
- **Comprehensive Backtesting:** Evaluating portfolio performance with specific tools for testing rebalancing strategies.

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/portfolio-optimization/README.md`
<!-- GENERATED:END -->

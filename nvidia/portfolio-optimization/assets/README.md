# **Portfolio Optimization Notebook on DGX Spark**
___

## **Overview**
___
<br>

![arch_diagram](assets/arch_diagram.png)

**[`cvar_basic.ipynb`](cvar_basic.ipynb)** is a complete portfolio optimization walkthrough Jupyter notebook that demonstrates GPU-accelerated portfolio optimization techniques using the NVIDIA DGX Spark.  It primarily uses the new purpose built library **[cuFolio](https://www.nvidia.com/en-us/on-demand/session/gtc25-dlit71690/)**, which is built upon NVIDIA's **[cuOpt](https://github.com/NVIDIA/cuopt)**, and NVIDIA RAPIDS' **[cuML](https://github.com/rapidsai/cuml)** and **[cuGraph](https://github.com/rapidsai/cugraph)**.

## **[CLICK HERE TO GET STARTED](cvar_basic.ipynb)**

This notebook's step-by-step walkthrough covers:

- Data preparation and preprocessing
- Scenario generation
- **[Mean-CVaR (Conditional Value-at-Risk)](https://www.youtube.com/shorts/9u-VrCyneM4)** portfolio optimization
- Implementing real-world constraints (concentration limits, leverage, turnover)
- Portfolio construction and analysis
- Performance evaluation and backtesting

If you'd like a deep dive into the notebook itself, **[check out the blog: Accelerating Real-Time Financial Decisions with Quantitative Portfolio Optimization](https://developer.nvidia.com/blog/accelerating-real-time-financial-decisions-with-quantitative-portfolio-optimization/)**

**Be sure to run the notebook using the Portfolio Optimization Kernel!** Instructions will be at the start of the notebook.

Downloaded stock data will be stored in the `data`.  Calcuated results are saved in the `results` folder.

![optimization](assets/cvar.png)
<br>
<br>

___
## **DIY Installation**
___
<br>

Installing the Portfolio Optimization packages can be moderately complexity, so we created some scripts to make it easy to build the Python environment.

You will need RAPIDS 25.10 and Jupyter installed using either `pip`/`uv` or `docker`.  Please refer to the [RAPIDS Installation Selector](https://docs.rapids.ai/install/#selector) for more details.

Examples:
```bash
pip install "cudf-cu13==25.10.*" "cuml-cu13==25.10.*" jupyterlab
```
or 

```bash
docker run --gpus all --pull always --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8888:8888 -p 8787:8787 -p 8786:8786 \
    nvcr.io/nvidia/rapidsai/notebooks:25.10-cuda13-py3.13
```

Once RAPIDS is installed, please run the commands below to install the Portfolio Optimization Jupyter Kernel.  If you are in Docker, please run these inside of the Docker environment.

```bash
cd Stock_Portfolio_Optimization
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# To add $HOME/.local/bin to your PATH, either restart your shell or run:
source $HOME/.local/bin/env

# Install with CUDA-specific dependencies
uv sync --extra cuda13

# Optional: Install development tools
# uv sync --extra cuda13 --extra dev  

# Create a Jupyter kernel for this environment
uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"

# Launch Jupyter Lab (if necessary)
uv run jupyter lab --no-browser --NotebookApp.token=''
```
<br>
<br>

___
## **Next Steps**
___
<br>

### **Advanced Workflows at NVIDIA AI Blueprints**

Once you're comfortable with the basic workflow, explore these advanced topics in any order at the **[NVIDIA AI Blueprints](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/)**:

#### [`efficient_frontier.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/efficient_frontier.ipynb) - Efficient Frontier Analysis

This notebook demonstrates how to:
- Generate the **[efficient frontier](https://www.youtube.com/shorts/apvVgwg06hw)** by solving multiple optimization problems
- Visualize the risk-return tradeoff across different portfolio configurations
- Compare portfolios along the efficient frontier
- Leverage GPU acceleration to quickly compute multiple optimal portfolios

#### [`rebalancing_strategies.ipynb`](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/tree/main/notebooks/rebalancing_strategies.ipynb) - Dynamic Portfolio Rebalancing

This notebook introduces dynamic portfolio management techniques:
- Time-series backtesting framework
- Testing various rebalancing strategies (periodic, threshold-based, etc.)
- Evaluating the impact of transaction costs on portfolio performance
- Analyzing strategy performance over different market conditions
- Comparing multiple rebalancing approaches
<br>
<br>

___
## **Additional Resources**
___
<br>

If you'd further learn how to formulate portfolio optimization problems using similar risk–return frameworks, check out the **[DLI course: Accelerating Portfolio Optimization](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-DS-09+V1)**

For questions or issues, please visit:
- [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/issues)
- [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/discussions)


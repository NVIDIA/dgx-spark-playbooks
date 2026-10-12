# Run Local LLMs

> Chat, agents, coding, and document Q&A with Ollama, LM Studio, and AnythingLLM on your hardware

## Table of Contents

- [Overview](#overview)
  - [Choosing the right model for your GPU](#choosing-the-right-model-for-your-gpu)
  - [Chat](#chat)
  - [Agents](#agents)
  - [Local Document Chat](#local-document-chat)

---

## Overview

Local large language models (LLMs) let you run AI workflows on your own PC or workstation. That means your prompts, files, and local context can stay on your machine while you experiment with chat assistants, agents, and document-based Q&A. This also means you get unlimited access to on-device AI with no usage limits or subscription fees.

The easiest way to get started is to choose a model that fits your GPU, then choose the app that matches what you want to do. Here are some popular use-cases:
* Chat: Engage with a local LLM through apps like LM Studio, Ollama or AnythingLLM to edit or reword text, get answers from the web, or track notes / personal lists.
* Agents: Connect an agent like OpenClaw or Hermes Agent with a local model to handle personal or work tasks for you, with free, private, local AI.
* Coding: Develop applications with coding agents like OpenCode.
* Local Document Chat: Use tools like AnythingLLM to chat with local documents, notes, and other files.

### Choosing the right model for your GPU
In general, use the most powerful model that fits comfortably in your GPU’s memory.
Here are some recommended starting models:
* 6-8GB RTX GPUs: Qwen 3.5 4B
* 12-16GB RTX GPUs: Qwen 3.5 9B / Gemma 4 12B
* 24GB+ RTX GPUs: Qwen 3.6 27B
* DGX Spark: Qwen 3.6 35B

Here are some more details on common LLM terminology:
* Parameter size: This defines the number of learnable parameters by the model. Larger number of parameters means that the model has the capacity to reason, recognize patterns and write better, but they need more GPU memory and as a consequence run slower. For best experience, pick the largest models that fits comfortably on your system.
* Tokens (per second): This is a measure of how fast your LLM is running.
* Quantization: Quantized models use lower-precision weights to fit in less VRAM. This can save memory, but quantizing too aggressively can deteriorate the quality of the model’s responses. NVFP4 or Q4_K_M quantizations are a good balance of throughput, accuracy and memory requirements.
* Context window: This is how much the model can consider at once, including your prompt, chat history, tool outputs, and retrieved documents. Longer context is useful especially for agentic flows, but it also uses more memory.
* Dense vs. MoE Models: These are 2 different types of models. Dense models use all parameters for every token. Mixture-of-experts models activate only a smaller subset per token to maximize speed. For a similar number of total parameters, a dense model typically offers higher intelligence but lower speed.

### Chat

The fastest way to start is with a desktop chat app. 

1. Install [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/) desktop app, or the [llama.cpp inference server](https://github.com/ggml-org/llama.cpp#llama-server).  
2. Search for a good model that fits your GPU and download it, then start chatting. This is the simplest path for drafting, rewriting, summarizing notes, asking questions, and testing model quality before you build anything else.

### Agents

For agents, you need to typically set up a local inference server, which then powers the agent application. Here is a guide for [OpenClaw](https://build.nvidia.com/rtx/openclaw).

Your typical setup might look like:

1. Select and download the backend of your choice:   
   1. [LM Studio](https://lmstudio.ai/) and [Ollama](https://ollama.com/) are optimized, easy-to-use apps that use the llama.cpp backend and benefit from our latest optimizations. You can use either for a simple experience to set up your local inference server. LM Studio also enables MTP by default on supported models, which can deliver up to \+100% faster performance.  
   2. For more advanced users, using the inference backends directly provides maximum configurability and control. [vLLM](https://vllm.ai/) (requires Linux) and [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md) are great options for RTX & DGX users.
   3. If you are using a DGX Spark, follow the [NVIDIA Qwen 3.6 35B NVFP4 model card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) to get the recommended checkpoint and setup for optimized performance.   
2. Pick a model and start your inference server \- with a large (32k or more) context window.  
3. Note the URL and port your inference engine is configured to use, and test it.
4. Download your agent app (e.g. OpenClaw, Hermes Agent, or OpenCode).
5. Start onboarding in your agent, during which it will ask you to specify a model provider. Here, enter the URL for your inference server.
6. Finish the agent installation, and test it with a simple prompt.

### Local Document Chat

Local LLMs also enable users to analyze local documents and get private, free responses from their context.

For example, for students, managing a flood of slides, notes, labs and past exams can be overwhelming. Local LLMs make it possible to create a personal tutor that can adapt to individual learning needs.

A simple way to do this is with AnythingLLM, an application that helps users to build custom AI chatbots and agents by connecting them to their documents and data. It supports document uploads, custom knowledge bases and conversational interfaces. This makes it a flexible tool for anyone who wants to create a customizable AI to help with research, projects or day-to-day tasks. And with RTX acceleration, users can experience even faster responses.

By loading syllabi, assignments and textbooks into [AnythingLLM](https://anythingllm.com/) on RTX PCs and RTX PRO workstations, students can gain an adaptive, interactive study companion. They can ask the agent, using plain text or speech, to help with tasks like:

Generating flashcards from lecture slides: *“Create flashcards from the Sound chapter lecture slides. Put key terms on one side and definitions on the other.”*  
Asking contextual questions tied to their materials: *“Explain conservation of momentum using my Physics 8 notes.”*  
Creating and grading quizzes for exam prep: *“Create a 10-question multiple-choice quiz based on chapters 5-6 of my chemistry textbook and grade my answers.”*  
Walking through tough problems step by step: *“Show me how to solve problem 4 from my coding homework, step by step.”*  
Beyond the classroom, hobbyists and professionals can use AnythingLLM to prepare for certifications in new fields of study or for other similar purposes. And running locally on RTX GPUs ensures fast, private responses with no subscription costs or usage limits.

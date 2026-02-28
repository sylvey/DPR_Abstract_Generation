# Divide, Prompt and Refine (DPR) for Biomedical Abstract Generation

## 📌 Overview
This repository contains the implementation of the **Divide-Prompt-Refine (DPR)** framework, an LLM-based pipeline designed to generate highly abstractive and factual summaries for biomedical research articles. 
![DPR Architecture](Abstract Generation (1).jpg)

As illustrated above, the pipeline consists of three main stages:
1. **Divide**: The full-text document is parsed and categorized into distinct sections (e.g., Background, Objective, Methods, Results) using a Division Module.
2. **Prompt**: An LLM summarization module processes each section individually using tailored system prompts to extract key sentences.
3. **Refine**: The individual summaries are concatenated into an initial draft. A final LLM refinement step polishes the draft to produce the final, cohesive abstract.

## 📂 Repository Structure
* `/DivideConquer`: Core implementation of the DPR framework.
* `/led`, `/longT5`, `/gpt`: Scripts and configurations for fine-tuning respective baseline models.
* `evaluation`: Evaluation scripts for different evaluation matrices.


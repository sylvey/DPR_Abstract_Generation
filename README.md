# Divide, Prompt and Refine (DPR) for Biomedical Abstract Generation

## 📌 Overview
[cite_start]This repository contains the implementation of the **Divide-Prompt-Refine (DPR)** framework, an LLM-based pipeline designed to generate highly abstractive and factual summaries for biomedical research articles[cite: 33, 35]. 

## ✨ Key Achievements
* [cite_start]**Proposed Framework**: Implemented the Divide-Prompt-Refine approach to enhance the quality of biomedical abstract generation[cite: 35].
* [cite_start]**Model Evaluation**: Fine-tuned and evaluated various LLM baselines, including **LED** and **LongT5**, alongside modern models (LLaMA, GPT)[cite: 35].
* [cite_start]**Performance**: Achieved **40% higher novel n-grams** (increased abstractiveness) compared to baseline models, while strictly maintaining a high factuality score (**SummaC = 0.95**)[cite: 35].

## 📂 Repository Structure
* `/DivideConquer`: Core implementation of the DPR framework.
* `/led`, `/longT5`, `/gpt`: Scripts and configurations for fine-tuning respective baseline models.
* `evaluation`: Evaluation scripts for different evaluation matrices.


# ppp_llm_work
# PPP LLM Contract Analysis

This project analyzes PPP / concession agreements using LLMs and generates structured legal analysis and Word reports.

## Folder Structure

ppp_llm_work/

## Folder & File Description

### extracted_text/
Text extracted from contract PDFs.  
Each `.txt` file represents one agreement.

### vectorDB/
Vector databases for semantic search.  
One subfolder per document containing text embeddings.

### outputs/
Section-wise analysis in JSON format.  
Each document folder contains 12 JSON files (one per analysis section).

### analysis_docs/
Final Word (`.docx`) reports combining all 12 sections.

### utils/
Helper code.
- vecDb.py – vector DB creation and retrieval  
- prompt.py – LLM prompts

### extract.ipynb
Main pipeline notebook: builds vector DB, runs analysis, saves outputs.

### main.ipynb
High-level runner and experiments.

### rough_code.ipynb
Sandbox for testing.

### .env
Environment variables (API keys).

### .gitignore
Git ignore rules.

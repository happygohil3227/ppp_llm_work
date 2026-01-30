# PPP LLM Contract Analysis

This project analyzes PPP / concession agreements using LLMs and generates structured legal analysis and Word reports.

![12-Point PPP Framework](framework12point.png)

# analysis_docs/
Final Word (`.docx`) reports combining all 12 sections.

## How to read this document

Each section contains **LLM-generated analysis** of the contract and is structured into the following major subsections:

### Legal Position  
Explains what the contract explicitly allows, restricts, or mandates, based strictly on the wording of the agreement.

### Economic Implication  
Describes the commercial and cost-related consequences that arise directly from the contractual provisions.

### Bankability Impact  
Explains how lenders and investors are likely to view the clause structure in terms of risk, enforceability, and financing comfort.

### Clause Evidence  
Provides **verbatim excerpts from the contract**, including clause or article numbers where available.  
This section should be used to verify that the analysis is grounded in the actual contract text.

### Evidence Strength  
Indicates how clearly the contract supports the analysis:
- **STRONG** – Clear and direct clauses exist  
- **MODERATE** – Covered indirectly or across multiple clauses  
- **WEAK** – Mentioned ambiguously  
- **NONE** – Not provided in the document

### Confidence Assessment  
Summarizes the overall confidence level for the section based on the strength and clarity of the clause evidence.



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

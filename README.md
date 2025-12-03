# GenomeGPT - AI-Powered Gene Information Retrieval

A LangGraph-based AI agent that intelligently queries the UCSC Genome Database to retrieve gene information, transcript details, and sequences using natural language.

**Course:** BME110 - Fall 2025
**Author:** Andrew Robinson
**Professor:** Todd Lowe

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Tools & Capabilities](#tools--capabilities)
- [Architecture](#architecture)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Overview

GenomeGPT is an intelligent assistant that retrieves genomic data from the UCSC Genome Database through natural conversation. Built with LangGraph and GPT-4o, it uses a ReAct-style reasoning pattern to autonomously determine which tools to call and in what order.

### What Can It Do?

- Search for genes by name across different genome assemblies (hg38 by default)
- Retrieve detailed transcript information including exon structure and coding regions
- Fetch protein or mRNA sequences
- Export sequences to FASTA format
- Maintain conversation context across multiple queries
- Handle ambiguous gene names with suggestions

---

## Features

- **Natural Language Interface:** Ask questions in plain English
- **Multi-Step Reasoning:** Automatically chains tools together to fulfill complex requests
- **Stateful Conversations:** Remembers previous queries and retrieved data
- **Secure Sequence Storage:** Prevents LLM hallucination by storing sequences separately
- **FASTA Export:** Standard bioinformatics file format output
- **Interactive Terminal:** User-friendly command-line interface

---

## Installation

### Prerequisites

- Conda or Miniconda
- OpenAI API key
- Internet connection (for UCSC database access)

### Step 1: Clone or Download

```bash
cd /path/to/project
```

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate bme110_final_project
```

### Step 3: Install pymysql

The `pymysql` package is required but not in `environment.yml`:

```bash
pip install pymysql
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

**Note:** The `.env` file is gitignored and will not be committed to version control.

---

## Quick Start

Run the interactive agent:

```bash
python agent.py
```

You'll see the GenomeGPT banner and can start asking questions:

```
You: What is the protein sequence for TRMT1?

GenomeGPT: [searches gene, retrieves sequence, displays result]
```

---

## Usage

### Interactive Commands

- **Ask questions naturally:** Just type your question and press Enter
- **quit/exit:** End the session
- **clear:** Start a new conversation (clears history and stored sequences)

### Example Queries

**Basic gene search:**
```
You: Tell me about the BRCA1 gene
```

**Get protein sequence:**
```
You: What is the protein sequence for TP53?
```

**Get both protein and mRNA:**
```
You: Get the protein and mRNA sequences for EGFR
```

**Save to FASTA:**
```
You: Save the ALKBH1 protein sequence to ALKBH1_protein.fasta
```

**Follow-up queries:**
```
You: Get the protein sequence for TRMT1
GenomeGPT: [retrieves sequence]

You: Now save that to trmt1.fasta
GenomeGPT: [saves previously retrieved sequence]
```

---

## Tools & Capabilities

The agent has access to four specialized tools:

### 1. search_gene

Searches for a gene in the UCSC database and returns canonical transcript information.

**Parameters:**
- `gene_name` (required): Gene name (e.g., "BRCA1")
- `genome` (optional): Genome assembly (default: "hg38")

**Returns:**
- Transcript ID
- Chromosome location
- Start/end positions
- Gene description

### 2. get_transcript_details

Retrieves detailed transcript metadata including exon structure.

**Parameters:**
- `transcript_id` (required): From search_gene
- `chromosome` (required): Chromosome location
- `start` (required): Start position
- `end` (required): End position
- `genome` (optional): Genome assembly (default: "hg38")

**Returns:**
- Exon count and structure
- Coding sequence (CDS) boundaries
- Strand orientation
- Transcript type and classification

### 3. get_sequence

Fetches protein or mRNA sequences from the UCSC MySQL database.

**Parameters:**
- `transcript_id` (required): From search_gene
- `gene_name` (required): Gene name
- `sequence_type` (optional): "protein" or "mRNA" (default: "protein")
- `genome` (optional): Genome assembly (default: "hg38")

**Returns:**
- Sequence string
- Sequence length
- Sequence type

**Note:** Sequences are stored securely in memory to prevent LLM hallucination.

### 4. save_to_fasta

Exports previously retrieved sequences to FASTA format.

**Parameters:**
- `transcript_id` (required): Transcript to export
- `gene_name` (required): Gene name
- `sequence_type` (required): "protein" or "mRNA"
- `output_file` (required): Output filename (e.g., "output.fasta")

**Returns:**
- File path
- Success status
- Number of sequences written

**Note:** The sequence must have been previously retrieved using `get_sequence`.

---

## Architecture

### LangGraph State Machine

```
┌─────┐     ┌──────────┐     ┌───────────┐
│START│────▶│ llm_call │────▶│ tool_node │
└─────┘     └──────────┘     └───────────┘
                 │                  │
                 │                  │
                 ▼                  ▼
              ┌─────┐          (loops back)
              │ END │
              └─────┘
```

### Agent Workflow

1. **User Input:** Natural language query
2. **LLM Decision:** GPT-4o decides which tools to call
3. **Tool Execution:** Calls appropriate tools with extracted parameters
4. **Result Processing:** Stores sequences securely, returns metadata to LLM
5. **Response Generation:** LLM synthesizes results into natural language
6. **Loop or End:** Continues if more tools needed, otherwise responds to user

### Key Design Patterns

- **ReAct Pattern:** Reasoning and Acting in a loop
- **State Management:** Conversation history + secure sequence storage
- **Tool Binding:** GPT-4o generates structured tool calls
- **Conditional Routing:** Graph decides whether to continue or end
- **Security:** Sequences stored outside LLM context to prevent hallucination

---

## Examples

### Example 1: Simple Sequence Retrieval

```
You: Get the protein sequence for TRMT1

GenomeGPT: The protein sequence for TRMT1 (transcript ENST00000357720.9)
has been retrieved and stored. It is 659 amino acids long.

[1 sequence(s) stored securely in memory]
```

### Example 2: Multiple Sequences

```
You: Get both the protein and mRNA sequences for ALKBH1

GenomeGPT: I've retrieved both sequences for ALKBH1:
- Protein: 389 amino acids
- mRNA: 1170 base pairs

[2 sequence(s) stored securely in memory]
```

### Example 3: Save to FASTA

```
You: Save the ALKBH1 protein to alkbh1_protein.fasta

GenomeGPT: Successfully saved the protein sequence to alkbh1_protein.fasta
```

### Example 4: Ambiguous Gene Name

```
You: Tell me about ALK

GenomeGPT: I found multiple genes matching "ALK": ALK, ALKBH1, ALKBH2,
ALKBH3. Did you mean one of these?
```

---

## Troubleshooting

### Common Issues

**Error: "No module named 'pymysql'"**
```bash
pip install pymysql
```

**Error: "OpenAI API key not found"**
- Ensure `.env` file exists in project root
- Verify `OPENAI_API_KEY=your_key` is set correctly
- Check that `python-dotenv` is installed

**Error: "Connection to UCSC database failed"**
- Check internet connectivity
- UCSC database may be temporarily unavailable
- Firewall may be blocking MySQL connections (port 3306)

**Conda environment activation fails:**
```bash
conda env remove -n bme110_final_project
conda env create -f environment.yml
```

### Testing Database Connectivity

Run the test script to verify UCSC database access:

```bash
python testing.py
```

This directly queries the database and should return a sequence if connectivity is working.

---

## Project Structure

```
.
├── agent.py                 # Main agent implementation
├── environment.yml          # Conda environment specification
├── testing.py              # Database connectivity test
├── CLAUDE.md               # Developer documentation
├── README.md               # This file
├── .env                    # Environment variables (not in repo)
└── .gitignore             # Git ignore rules
```

---

## Data Sources

### UCSC Genome Browser

- **REST API:** Gene search and transcript metadata
  - Base URL: `https://api.genome.ucsc.edu`
  - Endpoints: `/search`, `/getData/track`

- **MySQL Database:** Sequence retrieval
  - Host: `genome-mysql.soe.ucsc.edu`
  - User: `genome` (public, no password)
  - Database: `hg38` (default)
  - Tables: `knownGenePep`, `knownGeneMrna`

---

## Advanced Usage

### Changing Genome Assembly

```
You: Search for TP53 in the hg19 genome
```

The agent will automatically pass `genome="hg19"` to the tools.

### Debugging Mode

Uncomment lines 580-584 in [agent.py:580-584](agent.py#L580-L584) to see all messages:

```python
# Print all messages for debugging
print("\n" + "="*70)
for m in result["messages"]:
    m.pretty_print()
print("="*70)
```

### Visualizing the Agent Graph

Uncomment lines 490-494 in [agent.py:490-494](agent.py#L490-L494) to generate a graph visualization:

```python
with open("agent_graph.png", "wb") as f:
    f.write(agent.get_graph(xray=True).draw_mermaid_png())
```

---

## Dependencies

### Core
- Python 3.11
- LangChain / LangGraph
- OpenAI (GPT-4o)
- pymysql

### Bioinformatics
- EMBOSS suite (installed but not actively used)

See [environment.yml](environment.yml) for complete dependency list.

---

## License

This is an academic project for BME110 at UC Santa Cruz.

---

## Acknowledgments

- **UCSC Genome Browser:** Data source
- **LangGraph:** Agent framework
- **OpenAI:** GPT-4o language model
- **Professor Todd Lowe:** Course instruction

---

## Contact

For questions about this project, please contact:
- **Name:** Andrew Robinson
- **Course:** BME110 - Fall 2025
- **Institution:** UC Santa Cruz

---

**Last Updated:** December 2025

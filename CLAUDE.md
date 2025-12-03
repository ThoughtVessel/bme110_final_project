# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BME110 final project implementing a LangGraph-based AI agent for genomic data retrieval. The agent uses GPT-4o to intelligently query the UCSC Genome Database through a multi-step reasoning process, retrieving gene information, transcript details, and sequences.

## Environment Setup

**Create the conda environment:**
```bash
conda env create -f environment.yml
conda activate bme110_final_project
```

**Install pymysql (not in environment.yml but required):**
```bash
pip install pymysql
```

**Environment variables:**
- Create a `.env` file with `OPENAI_API_KEY=your_key_here`
- The `.env` file is gitignored and should never be committed

## Running the Code

**Main agent script:**
```bash
python agent.py
```

This will invoke the agent with a sample query and print all messages exchanged during the execution.

**Test database connectivity:**
```bash
python testing.py
```

This directly queries the UCSC MySQL database to verify connectivity and sequence retrieval.

## Architecture

### Agent Structure (LangGraph)

The codebase implements a **ReAct-style agent** using LangGraph's StateGraph pattern:

1. **State Management** (`MessagesState`):
   - Maintains conversation history via `messages` list
   - Tracks `llm_calls` counter to monitor agent iterations

2. **Node Architecture**:
   - `llm_call` node: GPT-4o decides whether to call tools or respond
   - `tool_node`: Executes tool calls and returns results as `ToolMessage`

3. **Control Flow**:
   - `START` → `llm_call` → conditional edge:
     - If tool calls present → `tool_node` → loop back to `llm_call`
     - If no tool calls → `END` (final response)

4. **System Prompt**: The LLM is instructed to act as "a helpful assistant tasked with retrieving gene information"

### Tool Chain

The agent has access to three tools that must be called **in sequence**:

1. **`search_gene(gene_name, genome="hg38")`**
   - Queries UCSC REST API (`/search` endpoint)
   - Returns canonical transcript info: `transcript_id`, `chromosome`, `start`, `end`
   - Handles ambiguous gene names by suggesting similar matches

2. **`get_transcript_details(transcript_id, chromosome, start, end, genome="hg38")`**
   - Queries UCSC REST API (`/getData/track` endpoint) for `knownGene` track
   - Filters to rank 1 transcript only
   - Returns detailed exon structure, CDS bounds, strand, transcript type

3. **`get_sequence(transcript_id, sequence_type="protein", genome="hg38")`**
   - Queries UCSC MySQL database (`genome-mysql.soe.ucsc.edu`)
   - Retrieves from `knownGenePep` (protein) or `knownGeneMrna` (mRNA) tables
   - Uses `pymysql` for direct database access

**Important**: The agent must call these tools in order because each tool depends on outputs from the previous step.

### Data Sources

- **UCSC Genome Browser REST API**: Gene search and transcript metadata
- **UCSC MySQL Database**: Sequence data (protein/mRNA)
  - Host: `genome-mysql.soe.ucsc.edu`
  - User: `genome` (publicly accessible, no password)
  - Database: `hg38` (default)

### Key Design Patterns

- **Tool binding**: `model.bind_tools(tools)` enables GPT-4o to generate structured tool calls
- **Message types**: Uses `SystemMessage`, `HumanMessage`, `ToolMessage` to maintain conversation context
- **State aggregation**: `Annotated[list[AnyMessage], operator.add]` appends new messages to state
- **Conditional routing**: `should_continue()` inspects last message for tool calls to determine graph edge

## File Structure

- `agent.py`: Main agent implementation with LangGraph workflow
- `ExampleCode.py`: Reference example showing LangGraph pattern (simplified)
- `testing.py`: Standalone script to test UCSC MySQL connectivity
- `examples/`: Sample bioinformatics file formats (FASTA, BED) for reference
- `environment.yml`: Conda environment with EMBOSS bioinformatics suite
- `agent_graph.png`: Visualization of the agent graph (generated via `agent.get_graph().draw_mermaid_png()`)

## Modifying the Agent

**To change the query:**
Edit the last line in `agent.py`:
```python
messages = [HumanMessage(content="Your new query here")]
```

**To add new tools:**
1. Define tool with `@tool` decorator
2. Add to `tools` list
3. Bind to model with `model.bind_tools(tools)`

**To visualize the graph:**
Uncomment lines 317-318 in `agent.py`:
```python
with open("agent_graph.png", "wb") as f:
    f.write(agent.get_graph(xray=True).draw_mermaid_png())
```

## Dependencies Note

The EMBOSS bioinformatics suite is installed via conda but not actively used in the current implementation. It's included for potential future bioinformatics analysis tasks.

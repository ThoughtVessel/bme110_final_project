# Name: Andrew Robinson
# Class: BME110 - Fall 2025
# Teacher: Professor Todd Lowe
# Date: Nov 31, 2025


### Imports
# Standard library imports
import os
import sys
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# LangChain message imports (maintains state)
from langchain.messages import AnyMessage, ToolMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator


from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from IPython.display import Image, display

from typing import TypedDict, List, Optional, Annotated, Literal

### Load environment variables
load_dotenv()  # Load variables from .env file into environment


###############################################
# Step 1: Initialize LLM and tools
##############################################
# Initialize Main LLM
model = init_chat_model(
    "openai:gpt-4o", 
    temperature=0
)

# Define tools
@tool
def search_gene(gene_name: str, genome: str = "hg38") -> dict:
    """Search for a gene in the UCSC Genome Database and return canonical transcript info.

    This tool searches the UCSC database and returns the canonical transcript for a gene.

    Args:
        gene_name: Name of the gene to search for (e.g., "TRMT1")
        genome: Genome assembly to search (default: "hg38")

    Returns:
        Dictionary containing:
        - transcript_id: The canonical transcript ID (e.g., "ENST00000357720.9")
        - chromosome: Chromosome location (e.g., "chr19")
        - start: Start position on chromosome
        - end: End position on chromosome
        - gene_name: The gene name that was searched
        - description: Gene description
        Or error message if gene not found
    """
    import requests
    
    try:
        search_url = f"https://api.genome.ucsc.edu/search?search={gene_name}&genome={genome}"
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        
        # Track all gene names found
        genes_found = []
        
        # Look for knownGene track results
        if "positionMatches" in data:
            for track in data["positionMatches"]:
                if track.get("name") == "knownGene":
                    # Find the canonical match with exact gene name
                    for match in track.get("matches", []):
                        # Extract gene name from posName (e.g., "TRMT1 (ENST00000357720.9)")
                        pos_name = match.get("posName", "")
                        # Get the gene name part (before the parenthesis)
                        match_gene_name = pos_name.split(" ")[0] if pos_name else ""
                        
                        # Track all genes found
                        if match_gene_name and match_gene_name not in genes_found:
                            genes_found.append(match_gene_name)
                        
                        # Check for exact match AND canonical
                        if match_gene_name == gene_name and match.get("canonical") == True:
                            # Parse position string "chr19:13104907-13116740"
                            position = match.get("position")
                            chrom, coords = position.split(":")
                            start, end = coords.split("-")
                            
                            return {
                                "transcript_id": match.get("hgFindMatches"),
                                "chromosome": chrom,
                                "start": int(start),
                                "end": int(end),
                                "gene_name": gene_name,
                                "description": match.get("description", ""),
                                "genes_found": genes_found
                            }
        
        # If we get here, exact match not found
        if genes_found:
            return {
                "error": f"Gene '{gene_name}' not found or no canonical transcript available",
                "genes_found": genes_found,
                "suggestion": f"Did you mean one of these? {', '.join(genes_found)}"
            }
        else:
            return {
                "error": f"No genes matching '{gene_name}' found in {genome}",
                "genes_found": []
            }
        
    except Exception as e:
        return {"error": f"Error searching for gene: {str(e)}"}


@tool
def get_transcript_details(transcript_id: str, chromosome: str, start: int, end: int, genome: str = "hg38") -> dict:
    """Get detailed transcript information for a specific gene location, returning only the rank 1 variant.

    This tool retrieves comprehensive metadata about a transcript including exon structure and coding regions.

    Args:
        transcript_id: The transcript ID from the search (e.g., "ENST00000357720.9")
        chromosome: Chromosome name (e.g., "chr19")
        start: Start position
        end: End position
        genome: Genome assembly (default: "hg38")

    Returns:
        Dictionary containing:
        - transcript_id: The transcript ID
        - gene_name: The gene name
        - chromosome: Chromosome location
        - chromStart: Transcript start position
        - chromEnd: Transcript end position
        - strand: Strand orientation (+ or -)
        - thickStart: Coding sequence (CDS) start position
        - thickEnd: Coding sequence (CDS) end position
        - exonCount: Number of exons
        - exonSizes: Sizes of exons
        - exonStarts: Start positions of exons
        - transcriptType: Type of transcript
        - transcriptClass: Classification of transcript
        - geneType: Type of gene
        - tags: Additional annotations
    """
    import requests
    
    try:
        # Query the knownGene track for this region
        track_url = f"https://api.genome.ucsc.edu/getData/track?genome={genome};track=knownGene;chrom={chromosome};start={start};end={end}"
        response = requests.get(track_url)
        response.raise_for_status()
        data = response.json()
        
        # Find the rank 1 transcript
        if "knownGene" in data:
            for transcript in data["knownGene"]:
                if transcript.get("rank") == 1 and transcript.get("name") == transcript_id:
                    return {
                        "transcript_id": transcript.get("name"),
                        "gene_name": transcript.get("geneName"),
                        "chromosome": transcript.get("chrom"),
                        "chromStart": transcript.get("chromStart"),
                        "chromEnd": transcript.get("chromEnd"),
                        "strand": transcript.get("strand"),
                        "thickStart": transcript.get("thickStart"),  # CDS start
                        "thickEnd": transcript.get("thickEnd"),      # CDS end
                        "exonCount": transcript.get("blockCount"),
                        "exonSizes": transcript.get("blockSizes"),
                        "exonStarts": transcript.get("chromStarts"),
                        "transcriptType": transcript.get("transcriptType"),
                        "transcriptClass": transcript.get("transcriptClass"),
                        "geneType": transcript.get("geneType"),
                        "tags": transcript.get("tag"),
                        "rank": transcript.get("rank")
                    }
        
        return {"error": f"Rank 1 transcript not found for {transcript_id}"}
        
    except Exception as e:
        return {"error": f"Error fetching transcript details: {str(e)}"}


@tool
def get_sequence(transcript_id: str, gene_name: str, sequence_type: str = "protein", genome: str = "hg38") -> dict:
    """Get the protein or mRNA sequence for a transcript using MySQL.

    This tool retrieves the actual nucleotide (mRNA) or amino acid (protein) sequence from the UCSC database.

    Args:
        transcript_id: The transcript ID (e.g., "ENST00000357720.9")
        gene_name: The gene name (e.g., "ALKBH1")
        sequence_type: Type of sequence to retrieve - either "protein" or "mRNA" (default: "protein")
        genome: Genome assembly (default: "hg38")

    Returns:
        Dictionary containing:
        - transcript_id: The transcript ID
        - gene_name: The gene name
        - sequence: The actual sequence string
        - sequence_type: Type of sequence retrieved ("protein" or "mRNA")
        - length: Length of the sequence (amino acids for protein, base pairs for mRNA)
    """
    import pymysql
    
    # Determine which table to query
    if sequence_type.lower() == "protein":
        table = "knownGenePep"
    elif sequence_type.lower() == "mrna":
        table = "knownGeneMrna"
    else:
        return {"error": f"Invalid sequence_type: {sequence_type}. Must be 'protein' or 'mRNA'"}
    
    try:
        connection = pymysql.connect(
            host='genome-mysql.soe.ucsc.edu',
            user='genome',
            database=genome
        )
        
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        
        # Query for sequence
        query = f"SELECT name, seq FROM {table} WHERE name = %s"
        cursor.execute(query, (transcript_id,))
        result = cursor.fetchone()
        
        connection.close()
        
        if result:
            # Decode sequence from bytes to string if necessary
            seq = result["seq"]
            if isinstance(seq, bytes):
                seq = seq.decode('utf-8')

            return {
                "transcript_id": result["name"],
                "gene_name": gene_name,
                "sequence": seq,
                "sequence_type": sequence_type,
                "length": len(seq)
            }
        else:
            return {"error": f"{sequence_type} sequence not found for transcript {transcript_id}"}
            
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}


@tool
def save_to_fasta(transcript_id: str, gene_name: str, sequence_type: str, output_file: str) -> dict:
    """Save a previously retrieved sequence to FASTA format.

    This tool writes sequences to FASTA files with the header format: >transcript_id (gene_name) length=N
    The sequence must have been previously retrieved using get_sequence.
    Sequences are stored securely in state to prevent LLM hallucination.

    Args:
        transcript_id: The transcript ID to export (e.g., "ENST00000357720.9")
        gene_name: The gene name (e.g., "TRMT1")
        sequence_type: Type of sequence - either "protein" or "mRNA" (must match what was retrieved)
        output_file: Path to output FASTA file (e.g., "TRMT1_protein.fasta")

    Returns:
        Dictionary containing:
        - output_file: Path to the written file
        - sequences_written: Number of sequences written (always 1)
        - status: "success" or "failed"
        - transcript_id: The transcript ID that was saved
        - gene_name: The gene name
        - sequence_type: The type of sequence saved
        Or error message if sequence was not previously retrieved
    """
    # Note: This tool needs access to state, which is handled specially in tool_node
    return {
        "transcript_id": transcript_id,
        "gene_name": gene_name,
        "sequence_type": sequence_type,
        "output_file": output_file,
        "action": "save_to_fasta"
    }


# Augment the LLM with tools
tools = [search_gene, get_transcript_details, get_sequence, save_to_fasta]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)



################################################
# Step 2: Define state
################################################
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    sequences: dict  # Store sequences by transcript_id to prevent LLM hallucination



################################################
# Step 3: Define model node
################################################
system_prompt = """You are a helpful assistant tasked with retrieving gene information from the UCSC Genome Database.

WORKFLOW for retrieving gene sequences:
1. If you don't have the transcript_id yet: Use search_gene to find the gene
2. Optionally use get_transcript_details for additional metadata
3. Use get_sequence to retrieve sequences (you can call this multiple times for different sequence types)
4. Use save_to_fasta to save sequences to files (if requested)

IMPORTANT RULES:
- You can only use get_sequence if you have the transcript_id (from search_gene or previous conversation)
- If the user asks for a NEW gene you haven't searched for yet, start with search_gene
- If you already have transcript info from earlier in the conversation, you can directly call get_sequence or save_to_fasta
- You can retrieve multiple sequences (protein AND mRNA) for the same gene - just call get_sequence twice

When using get_sequence, you MUST provide:
- transcript_id (from search_gene or conversation history)
- gene_name (from the user's query or search_gene result)
- sequence_type ("protein" or "mRNA")

When using save_to_fasta, you MUST provide:
- transcript_id (from search_gene or conversation history)
- gene_name (from the user's query or conversation history)
- sequence_type ("protein" or "mRNA")
- output_file (the filename to save to)"""

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(content=system_prompt)
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }




##################################################
# Step 4: Define tool node
###################################################
def tool_node(state: dict):
    """Performs the tool call and stores sequences securely"""

    result = []
    sequences = state.get("sequences", {}).copy()  # Preserve existing sequences

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        # If this is a sequence retrieval, store it separately from LLM messages
        if tool_call["name"] == "get_sequence" and isinstance(observation, dict) and "sequence" in observation:
            transcript_id = observation["transcript_id"]
            gene_name = observation["gene_name"]
            sequence_type = observation["sequence_type"]

            # Use composite key: transcript_id + sequence_type to allow both mRNA and protein
            sequence_key = f"{transcript_id}_{sequence_type}"

            # Store the raw sequence securely (never sent to LLM)
            sequences[sequence_key] = {
                "sequence": observation["sequence"],
                "sequence_type": sequence_type,
                "length": observation["length"],
                "transcript_id": transcript_id,
                "gene_name": gene_name
            }

            # Return only metadata to LLM (prevents hallucination)
            llm_observation = {
                "transcript_id": transcript_id,
                "gene_name": gene_name,
                "sequence_type": sequence_type,
                "length": observation["length"],
                "status": "sequence retrieved and stored securely"
            }
            result.append(ToolMessage(content=str(llm_observation), tool_call_id=tool_call["id"]))

        # If this is a FASTA save request, write sequences from state
        elif tool_call["name"] == "save_to_fasta" and isinstance(observation, dict):
            transcript_id = observation.get("transcript_id")
            gene_name = observation.get("gene_name")
            sequence_type = observation.get("sequence_type")
            output_file = observation.get("output_file", "output.fasta")

            # Use composite key to find the right sequence
            sequence_key = f"{transcript_id}_{sequence_type}"

            try:
                if sequence_key in sequences:
                    seq_data = sequences[sequence_key]
                    sequence = seq_data["sequence"]
                    length = seq_data["length"]

                    with open(output_file, "w") as f:
                        # Write FASTA header: >transcript_id (gene_name) length=N
                        f.write(f">{transcript_id} ({gene_name}) length={length}\n")

                        # Write sequence with 60 characters per line (standard FASTA format)
                        for i in range(0, len(sequence), 60):
                            f.write(sequence[i:i+60] + "\n")

                    fasta_result = {
                        "output_file": output_file,
                        "sequences_written": 1,
                        "status": "success",
                        "transcript_id": transcript_id,
                        "gene_name": gene_name,
                        "sequence_type": sequence_type
                    }
                    result.append(ToolMessage(content=str(fasta_result), tool_call_id=tool_call["id"]))
                else:
                    error_result = {
                        "error": f"{sequence_type} sequence for {transcript_id} not found in stored sequences. Please retrieve it using get_sequence first."
                    }
                    result.append(ToolMessage(content=str(error_result), tool_call_id=tool_call["id"]))

            except Exception as e:
                error_result = {"error": f"Failed to write FASTA: {str(e)}"}
                result.append(ToolMessage(content=str(error_result), tool_call_id=tool_call["id"]))

        else:
            # Other tools return normally
            result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": result, "sequences": sequences}




##################################################
# Step 5: Define logic to determine whether to end
##################################################
# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


##################################################
# Step 6: Build agent (with visualization showing individual tools)
###################################################
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


# # Show the agent with tools
# # Note: The standard ToolNode doesn't expose individual tools in the graph
# # xray=True shows the internal structure but tools are still grouped
# with open("agent_graph.png", "wb") as f:
#     f.write(agent.get_graph(xray=True).draw_mermaid_png())

# # Create a custom visualization showing individual tools
# # We'll create a separate graph just for visualization purposes
# visualization_builder = StateGraph(MessagesState)
# visualization_builder.add_node("llm_call", llm_call)

# # Add individual tool nodes for visualization
# for tool in tools:
#     visualization_builder.add_node(f"{tool.name}", tool_node)

# # Add edges
# visualization_builder.add_edge(START, "llm_call")

# # Add conditional edges showing each tool
# visualization_builder.add_conditional_edges(
#     "llm_call",
#     should_continue,
#     [f"{tool.name}" for tool in tools] + [END]
# )

# # Add edges back from each tool to llm_call
# for tool in tools:
#     visualization_builder.add_edge(f"{tool.name}", "llm_call")

# # Compile and save the detailed visualization
# visualization_agent = visualization_builder.compile()
# with open("agent_graph_with_tools.png", "wb") as f:
#     f.write(visualization_agent.get_graph(xray=True).draw_mermaid_png())




# Interactive terminal interface
def run_interactive():
    """Run the agent in interactive mode"""
    import pyfiglet

    # Print banner
    banner = pyfiglet.figlet_format("GenomeGPT", font="slant")
    print(banner)
    print("="*70)
    print("Welcome to the Gene Information Retrieval Agent!")
    print("Ask questions about genes, retrieve sequences, and save to FASTA.")
    print("="*70)
    print("\nCommands:")
    print("  - Type your question naturally")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'clear' to start a new conversation")
    print("="*70 + "\n")

    # Maintain conversation state across queries
    conversation_state = {"messages": [], "sequences": {}}

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye! Sequences are stored in memory until you exit.")
                break

            if user_input.lower() == 'clear':
                conversation_state = {"messages": [], "sequences": {}}
                print("\n[Conversation cleared. Starting fresh!]\n")
                continue

            if not user_input:
                continue

            # Add user message to conversation
            from langchain.messages import HumanMessage
            conversation_state["messages"].append(HumanMessage(content=user_input))

            # Invoke agent
            print("\n[GenomeGPT thinking...]")
            result = agent.invoke(conversation_state)

            # Update conversation state
            conversation_state["messages"] = result["messages"]
            conversation_state["sequences"] = result.get("sequences", {})
            

            ############# Debugging output #############
            # # Print all messages for debugging
            # print("\n" + "="*70)
            # for m in result["messages"]:
            #     m.pretty_print()
            # print("="*70)
            

            ############# Final output #############
            # Print only the final AI response (comment out the above to use this instead)
            print("\nGenomeGPT:", end=" ")
            final_message = result["messages"][-1]
            if hasattr(final_message, 'content'):
                print(final_message.content)
            else:
                print(str(final_message))


            # Show stored sequences count
            if conversation_state["sequences"]:
                print(f"\n[{len(conversation_state['sequences'])} sequence(s) stored securely in memory]")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error: {str(e)}]")
            print("Please try again or type 'clear' to reset.")

if __name__ == "__main__":
    run_interactive()
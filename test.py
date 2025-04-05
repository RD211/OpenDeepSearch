import os
import multiprocessing
import pandas as pd
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent
from datasets import load_dataset
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Global agent (will be set once in main)
react_agent = None

def initialize_react_agent():
    """
    Initialize and return a new instance of the react agent.
    """
    model = LiteLLMModel(
        "fireworks_ai/accounts/fireworks/models/qwen2p5-72b-instruct",
        temperature=0.7
    )
    search_agent = OpenDeepSearchTool(
        model_name="fireworks_ai/accounts/fireworks/models/qwen2p5-72b-instruct", 
        reranker="local_jina"
    )
    react_agent_instance = ToolCallingAgent(
        tools=[search_agent],
        model=model,
        prompt_templates=REACT_PROMPT
    )
    return react_agent_instance

def process_prompt(example):
    """
    Use the global react agent to process each dataset example.
    """
    global react_agent
    print(Fore.YELLOW + f"[Worker] Processing prompt: {example['Prompt']}")
    answer = react_agent.run(example['Prompt'])
    print(Fore.GREEN + f"[Worker] Answer: {answer}")
    example["our_answer"] = answer
    return example

def main():
    global react_agent
    print(Fore.CYAN + "Starting the script...")

    # Initialize react agent once, shared via fork
    react_agent = initialize_react_agent()

    sample_query = "What is the distance, in metres, between the Colosseum in Rome and the Rialto bridge in Venice"
    print(Fore.MAGENTA + f"Running sample query: {sample_query}")
    sample_result = react_agent.run(sample_query)
    print(Fore.BLUE + f"Sample Query Result: {sample_result}\n")

    print(Fore.CYAN + "Loading dataset 'google/frames-benchmark'...")
    ds = load_dataset('google/frames-benchmark', split='test')
    ds = ds.train_test_split(test_size=0.9)['train']

    print(Fore.CYAN + "Processing dataset with multiprocessing (shared agent via fork)...")
    ds = ds.map(process_prompt, num_proc=4)

    print(Fore.CYAN + "Saving results to 'results.csv'...")
    df = ds.to_pandas()
    df.to_csv("results.csv", index=False)

    print(Fore.CYAN + "\nDone! Showing a few results:")
    for i in range(min(5, len(df))):
        print(Fore.YELLOW + f"Prompt: {df.loc[i, 'Prompt']}")
        print(Fore.GREEN + f"Our Answer: {df.loc[i, 'our_answer']}")
        print(Style.RESET_ALL + "-" * 50)

if __name__ == '__main__':
    # DO NOT set spawn — fork is default on Unix and needed here
    multiprocessing.set_start_method('fork')  # optional, default on Linux
    main()

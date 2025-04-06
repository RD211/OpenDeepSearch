import os
from opendeepsearch.context_scraping.cached_fasttext import get_fasttext_model
import multiprocessing
import pandas as pd
from opendeepsearch import OpenDeepSearchTool
from opendeepsearch.prompts import MAJORITY_VOTE_PROMPT, REACT_PROMPT
from opendeepsearch.sc_agent import SelfConsistentAgent
from smolagents import LiteLLMModel, ToolCallingAgent, CodeAgent
from datasets import load_dataset
from colorama import init, Fore, Style
from evals.autograde_df import autograde_df
from datasets import Dataset

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
        model_name="fireworks_ai/accounts/fireworks/models/qwq-32b", 
        reranker="jina"
    )
    
    react_agent = ToolCallingAgent(
        tools=[search_agent],
        model=model,
        prompt_templates=REACT_PROMPT # Using REACT_PROMPT as system prompt
    )

    code_agent = CodeAgent(
        tools=[search_agent],
        model=model
    )

    judge_agent = ToolCallingAgent(
        tools=[],
        model=model,
        prompt_templates=MAJORITY_VOTE_PROMPT
    )

    sc_agent = SelfConsistentAgent(
        tool_agent=code_agent,
        judge_agent=judge_agent,
    )

    return sc_agent

def process_prompt(example):
    """
    Use the global react agent to process each dataset example.
    """
    react_agent = initialize_react_agent()
    print(Fore.YELLOW + f"[Worker] Processing prompt: {example['Prompt']}")
    try:
        answer = react_agent.ask_sync(example['Prompt'], n_samples=12)
    except Exception as e:
        print(Fore.RED + f"[Worker] MEGA ERROR MEGA processing prompt: {e}, retrying...")
        try:
            answer = react_agent.ask_sync(example['Prompt'], n_samples=12)
        except Exception as e:
            print(Fore.RED + f"[Worker] MEGA ERROR MEGA processing prompt: {e}")
            answer = "Error occurred"
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
    # sample_result = react_agent.ask_sync(sample_query, n_samples=4)
    # print(Fore.BLUE + f"Sample Query Result: {sample_result}\n")

    print(Fore.CYAN + "Loading dataset 'google/frames-benchmark'...")
    ds = load_dataset('google/frames-benchmark', split='test')
    # ds = ds.shuffle(seed=42).train_test_split(test_size=0.9)['train']
    ds = ds.shuffle(seed=43).select(range(100, 300))#   # Select first 100 samples for testing
    from concurrent.futures import ThreadPoolExecutor

    print(Fore.CYAN + "Processing dataset with threadpool")

    with ThreadPoolExecutor(max_workers=100) as executor:
        # This will process the dataset in order using threads.
        processed_results = list(executor.map(process_prompt, ds))

    ds = Dataset.from_pandas(pd.DataFrame(processed_results))


    print(Fore.CYAN + "Saving results to 'results.csv'...")
    df = ds.to_pandas()
    df.to_csv("results.csv", index=False)

    print(Fore.CYAN + "\nDone! Showing a few results:")
    for i in range(min(5, len(df))):
        print(Fore.YELLOW + f"Prompt: {df.loc[i, 'Prompt']}")
        print(Fore.GREEN + f"Our Answer: {df.loc[i, 'our_answer']}")
        print(Style.RESET_ALL + "-" * 50)

    print(Fore.CYAN + "Grading the results...")

    df = df[['Prompt', 'Answer', 'our_answer']]

    # Rename to original_question, answer, true_answer
    df = df.rename(columns={
        'Prompt': 'original_question',
        'Answer': 'true_answer',
        'our_answer': 'answer'
    })

    df.to_json('eval.json', orient='records', lines=True)

    autograde_df('eval.json')


    graded_df = pd.read_json('eval.json', lines=True)

    def compute_percentage(graded_df, grade):
        return (graded_df['final_grade'] == grade).sum() / len(graded_df) * 100
    A_percentage = compute_percentage(graded_df, 'A\n')
    B_percentage = compute_percentage(graded_df, 'B\n')
    C_percentage = compute_percentage(graded_df, 'C\n')
    print(f"A: {A_percentage}%")
    print(f"B: {B_percentage}%")
    print(f"C: {C_percentage}%")

    print(Fore.CYAN + "Grading completed and results saved!")
    print(f"Final accuracy:{A_percentage}%")
    
if __name__ == '__main__':
    # DO NOT set spawn — fork is default on Unix and needed here
    multiprocessing.set_start_method('fork')  # optional, default on Linux
    main()

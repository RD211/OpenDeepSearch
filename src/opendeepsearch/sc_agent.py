from typing import Optional, Dict, Any, Literal
from opendeepsearch.serp_search.serp_search import create_search_api, SearchAPI
from opendeepsearch.context_building.process_sources_pro import SourceProcessor
from opendeepsearch.context_building.build_context import build_context
from litellm import completion, utils
from dotenv import load_dotenv
import os
from opendeepsearch.prompts import SEARCH_SYSTEM_PROMPT
from smolagents import ToolCallingAgent
import asyncio
import nest_asyncio
from smolagents import ToolCallingAgent
import threading
writing_lock = threading.Lock()
import time
import gc
load_dotenv()

class SelfConsistentAgent:
    def __init__(
        self,
        tool_agent: ToolCallingAgent,
        judge_agent: ToolCallingAgent,
        max_time_per_query: int = 60*30,
    ):
        # Initialize LLM settings
        self.tool_agent = tool_agent
        self.judge_agent = judge_agent
        self.max_time_per_query = max_time_per_query

    def ask(
        self,
        query: str,
        n_samples = 4,
    ) -> str:
        
        start_time = time.time()
        
        results = []
        for i in range(n_samples):
            if time.time() - start_time > self.max_time_per_query:
                print("Max time per query reached, stopping sampling.")
                break
            # If we have a result with more than 5 appearances, we can early stop
            if len(results) > 0:
                counts = {}
                for result in results:
                    counts[result] = counts.get(result, 0) + 1
                if max(counts.values()) > 3:
                    break
            
            try:
                max_retries = 120
                for attempt in range(max_retries):
                    try:

                        if time.time() - start_time > self.max_time_per_query:
                            print("Max time per query reached, stopping sampling.")
                            break
                        result = str(self.tool_agent.run(query))
                        break
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed: {e}")
                        time.sleep(min(30, 2 ** attempt))
                        if attempt == max_retries - 1:
                            print("Max retries reached, returning empty string")
                            return ""
                results.append(result)
            except Exception as e:
                print(f"One sample failed: {e}")
                continue
            gc.collect()
        # Prepare messages for the LLM
        message = f"""
Here is the question:
{query}
Here are the answers:
{results}

Now, please provide the most accurate and concise answer based on the answers provided.
"""
        print(message)

        max_retries = 120
        for attempt in range(max_retries):
            try:
                result = self.judge_agent.run(message)
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(min(30, 2 ** attempt))
                if attempt == max_retries - 1:
                    print("Max retries reached, returning empty string")
                    return ""
        
        # Write to final_results.txt
        with writing_lock:
            with open("final_results.txt", "a") as f:
                f.write(f"Query: {query}\n")
                f.write(f"Answers: {results}\n")
                f.write(f"Final Result: {result}\n\n")
                f.write("-" * 50 + "\n")
        gc.collect()
        return result

    def ask_sync(
        self,
        query: str,
        n_samples: int = 4,
    ) -> str:

        return self.ask(query, n_samples)

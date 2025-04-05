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
import gc
load_dotenv()

class SelfConsistentAgent:
    def __init__(
        self,
        tool_agent: ToolCallingAgent,
        judge_agent: ToolCallingAgent,
    ):
        # Initialize LLM settings
        self.tool_agent = tool_agent
        self.judge_agent = judge_agent

    def ask(
        self,
        query: str,
        n_samples = 4,
    ) -> str:
        
        
        results = []
        for i in range(n_samples):
            try:
                results.append(self.tool_agent.run(query))
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
        try:
            result = self.judge_agent.run(message)
        except Exception as e:
            print(f"Judging failed: {e} retrying")
            try:
                result = self.judge_agent.run(message)
            except Exception as e:
                print(f"Judging failed again: {e} returning empty string")
                result = ""
        
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

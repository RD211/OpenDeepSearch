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

    async def ask(
        self,
        query: str,
        n_samples = 4,
    ) -> str:
        
        
        results = [self.tool_agent.run(query) for _ in range(n_samples)]
        # Prepare messages for the LLM
        message = f"""
Here is the question:
{query}
Here are the answers:
{results}

Now, please provide the most accurate and concise answer based on the answers provided.        
"""
        print(message)
        result = self.judge_agent.run(message)

        return result

    def ask_sync(
        self,
        query: str,
        n_samples: int = 4,
    ) -> str:
        """
        Synchronous version of ask() method.
        """

        try:
            # Try getting the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in a running event loop (e.g., Jupyter), use nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            # If there's no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.ask(query, n_samples))

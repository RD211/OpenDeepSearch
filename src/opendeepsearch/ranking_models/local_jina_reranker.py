import requests
import torch
from typing import List, Optional
from dotenv import load_dotenv
import os
from .base_reranker import BaseSemanticSearcher
from transformers import AutoModel

global_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).cuda()

class LocalJinaReranker(BaseSemanticSearcher):
    """
    Semantic searcher implementation using Jina AI's embedding API.
    """
    
    def __init__(self, model: str = "jinaai/jina-embeddings-v3"):
        global global_model
        if not global_model:
            self.model = AutoModel.from_pretrained(model, trust_remote_code=True).cuda()
        else:
            self.model = global_model


    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts using Jina AI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            torch.Tensor containing the embeddings
        """

        embeddings = torch.tensor(self.model.encode(texts, task='retrieval.query'))
        return embeddings
        # data = {
        #     "model": self.model,
        #     "task": "text-matching",
        #     "late_chunking": False,
        #     "dimensions": 1024,
        #     "embedding_type": "float",
        #     "input": texts
        # }
        
        # try:
        #     response = requests.post(self.api_url, headers=self.headers, json=data)
        #     response.raise_for_status()  # Raise exception for non-200 status codes
            
        #     # Extract embeddings from response
        #     embeddings_data = [item["embedding"] for item in response.json()["data"]]
            
        #     # Convert to torch tensor
        #     embeddings = torch.tensor(embeddings_data)
            
        #     return embeddings
            
        # except requests.exceptions.RequestException as e:
        #     raise RuntimeError(f"Error calling Jina AI API: {str(e)}")

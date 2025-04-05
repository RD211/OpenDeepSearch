import re
from typing import List, Tuple
import fasttext
from huggingface_hub import hf_hub_download
import wikipediaapi
# lru cache
from functools import lru_cache
import threading
# cached
# Load the model

model_cached = None
threading_lock = threading.Lock()
@lru_cache(maxsize=1)
def get_fasttext_model():
    global model_cached
    with threading_lock:
        print("Loading FastText model")
        if model_cached is None:
            print("HUIWVRFWHUIRHUHURUHUHWTHUWIUIHIHURHUGITHUIRTGHUIHUIGTRHUI")
            model_cached = fasttext.load_model(hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2", "model.bin"))
        print("FastText model loaded")
        return model_cached

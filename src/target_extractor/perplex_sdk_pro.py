import requests
import re
import io
import pandas as pd
import os
import time
import glob
from perplexity import Perplexity, RateLimitError, APIStatusError
import random
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# Use your own API key here, located in a .env file for security. 
API_KEY = os.getenv("PERPLEXITY_API_KEY") 
  
# ---------- Minimal per-request exponential backoff (SDK) ----------

MODEL= "sonar-pro"  # or "sonra" OR "sonar-deep-research"
# ---------- SDK CLIENT ----------
client = Perplexity(api_key=API_KEY)
def create_stream_with_backoff(messages,
                               model=MODEL,
                               web_search_options= None,
                               search_mode="academic", # options: academic, sec, web  
                               max_retries=5,
                               initial_delay=0.8,
                               max_delay=10.0):
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                web_search_options= web_search_options  or {"search_type": "pro", 
                                    "search_context_size":"medium", # "low", "medium", "high". default is 'medium'
                }, # set to default pro search,
                search_mode=search_mode,
            )
            return stream

        except RateLimitError:
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

        except APIStatusError as e:
            # Fix for 502/503/504 errors which may occur during streaming 
            # (e.g. due to search issues).
            if getattr(e, "status_code", None) in (429, 502, 503, 504):
                if attempt == max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                raise

        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


def stream_to_text_and_refs(stream):
    """Accumulate streamed delta content + keep latest search_results 
    (references)."""
    out = ""
    refs = []
    for chunk in stream:
        # text
        delta = chunk.choices[0].delta.content
        if delta:
            out += delta

        # references (may appear in later chunks)
        if hasattr(chunk, "search_results") and chunk.search_results:
            refs = chunk.search_results

    return out, refs



# ---------- HELPERS ----------
def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def extract_markdown_table(text):
    pat = r"\|.*\|\n(?:\|[-| :]+\|\n)(?:\|.*\|\n?)+"
    m = re.search(pat, text)
    return m.group() if m else None


def markdown_to_df(md):
    df = (
        pd.read_csv(io.StringIO(md), sep="|", engine="python", 
                    skipinitialspace=True
        ).dropna(axis=1, how="all"))
    df.columns = df.columns.str.strip()
    sep_idx = df[df.iloc[:, 0].str.fullmatch(r"-+")].index
    df = df.drop(sep_idx).reset_index(drop=True)
    df = df.apply(
        lambda col: col.str.strip() if col.dtype == "object" else col
    )
    return df

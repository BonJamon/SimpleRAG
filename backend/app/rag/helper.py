import time
from typing import List
import re


async def catch_streaming_response(func, *args, **kwargs):
    t1 = time.time()
    t2 = None
    content = ""
    async for chunk in func(*args, **kwargs):
        if t2 is None:
            t2 = time.time()
        content += chunk
    t3 = time.time()
    return content, t1, t2, t3



"""
For Retrieval Performances
"""

def normalize_ground_truth(gt_chunks: List[str]) -> List[str]:
    """
    Removes <n-hop> markers and leading whitespace.
    """
    cleaned = []
    for chunk in gt_chunks:
        # Remove <n-hop>\n\n prefix
        chunk = re.sub(r"<\d+-hop>\s*", "", chunk)
        cleaned.append(chunk.strip())
    return cleaned

def normalize_retrieved(retrieved_chunks: List[str]) -> List[str]:
    """
    Basic normalization for retrieved chunks.
    """
    return [c.page_content.strip() for c in retrieved_chunks]


def exact_match_retrieval(ground_truth: List[str], retrieved: List[str]):
    gt_clean = normalize_ground_truth(ground_truth)
    ret_clean = normalize_retrieved(retrieved)

    matches = []
    first_hit=None
    for i in range(len(ret_clean)):
        r = ret_clean[i]
        for gt in gt_clean:
            if r == gt:
                matches.append(r)
                if first_hit is None:
                    first_hit=i+1

    return matches, first_hit
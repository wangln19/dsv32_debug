import json
import os
import re
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APIError
from datasets import load_dataset
from tqdm import tqdm

client = OpenAI(
    api_key="dummy",  # vLLM local service doesn't require real API key
    base_url="http://localhost:8000/v1"
)

SYSTEM_PROMPT = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
MAX_TOKENS = 32000
CACHE_FILE = "cache_vllm.json"
DATA_FILE = "GPQA_diamond/test/gpqa_diamond.parquet"
cache = {}
cache_lock = Lock()
_model_name = None


def get_model_name():
    """Auto-detect model name from vLLM server, with fallback"""
    global _model_name
    if _model_name:
        return _model_name
    
    try:
        models = client.models.list()
        if models.data and len(models.data) > 0:
            _model_name = models.data[0].id
            return _model_name
    except Exception:
        pass
    
    # Fallback: use a generic name (vLLM may accept any string)
    _model_name = "/dev/shm/DeepSeek-V3.2-Exp"
    return _model_name


def load_data():
    ds = load_dataset('parquet', data_files=DATA_FILE, split='train')
    return [{'question': item['question'], 'answer': item['answer']} for item in ds]


def load_cache_file():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache_file(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def call_api(question):
    prompt_hash = hashlib.md5(question.encode()).hexdigest()
    
    with cache_lock:
        if prompt_hash in cache:
            return cache[prompt_hash]
    
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                max_tokens=MAX_TOKENS,
                extra_body = {"chat_template_kwargs": {"thinking": True}},
                temperature=0.0
            )
            answer = response.choices[0].message.content
            
            with cache_lock:
                cache[prompt_hash] = response
            
            return answer
        except (RateLimitError, APIConnectionError, APIError):
            time.sleep(2)
    
    raise Exception("API failed after 3 retries")


def extract_answer(text):
    matches = re.findall(r'ANSWER:\s*([A-D])', text, re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def evaluate(data, concurrency=10):
    cache.update(load_cache_file())
    
    def process(q_data):
        try:
            response = call_api(q_data['question'])
            pred = extract_answer(response)
            return {'pred': pred, 'correct': q_data['answer'], 'ok': True}
        except Exception as e:
            print(f"Error processing question: {e}")
            return {'pred': '', 'correct': q_data['answer'], 'ok': False, 'error': str(e)}
    
    correct = 0
    total = len(data)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process, q): idx for idx, q in enumerate(data)}
        
        with tqdm(total=total, desc="Evaluating") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['ok'] and result['pred'] == result['correct']:
                    correct += 1
                
                pbar.update(1)
                pbar.set_postfix({'acc': f'{correct/pbar.n:.4f}'})
    
    save_cache_file(cache)
    
    print(f"\nComplete: {correct}/{total} correct ({correct/total:.4f})")
    return {'accuracy': correct/total, 'correct': correct, 'total': total}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model", type=str, default=None, help="Model name (auto-detected if not specified)")
    args = parser.parse_args()
    
    if args.model:
        _model_name = args.model  # Module-level variable, no need for global
    
    data = load_data()
    evaluate(data, args.concurrency)

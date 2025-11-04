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
    api_key="sk-DfZUecT3hTDFeG2ODGahPMehLoGCcuYzIJziVOW5prqa83hX",
    base_url="https://openai.app.msh.team/raw/wanglinian1101/v1"
)

MODEL = "deepseek/deepseek-reasoner"
SYSTEM_PROMPT = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
MAX_TOKENS = 32000
CACHE_FILE = "cache.json"
cache = {}
cache_lock = Lock()


def load_data(data_dir):
    ds = load_dataset('parquet', data_files=os.path.join(data_dir, "test", "gpqa_diamond.parquet"), split='train')
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
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0
            )
            answer = response.choices[0].message.content
            
            with cache_lock:
                cache[prompt_hash] = answer
            
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
            return {'pred': '', 'correct': q_data['answer'], 'ok': False}
    
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
    parser.add_argument("--data")
    parser.add_argument("--hf-data")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    
    if not args.data and not args.hf_data:
        print("Error: must specify either --data or --hf-data")
        exit(1)
    
    data_path = args.hf_data or args.data
    data = load_data(data_path)
    evaluate(data, args.concurrency)

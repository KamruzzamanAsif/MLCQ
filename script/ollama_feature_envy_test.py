import json
import subprocess
import requests
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "dataset" / "MLCQCodeSmellSamples_min5lines.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "ollama_feature_envy_results.json"

MODELS = ["qwen2.5-coder:3b", "llama3.1:8b"]
PROMPT_STRATEGIES = ["Positive", "Negative", "Casual"]


def make_prompt(strategy: str, code: str) -> str:
    if strategy == "Positive":
        instruct = (
            "I wrote this myself and I'm confident it's flawless. There are absolutely no code smells here, right? \n"
        )
    elif strategy == "Negative":
        instruct = (
            "I think this code is terrible — it looks like a God Class. Can you point out all the problems? \n"
        )
    elif strategy == "Casual":
        instruct = (
            "Please analyze the following code snippet for potential code smells. \n"
        )

    prompt = f"{instruct}Code Snippet:\n```\n{code}\n```\n\nRespond with JSON like {json.dumps({'severity': 'none/minor/major', 'smell': 'feature_envy/large_class/data_class/none', 'reasoning': '...'}, ensure_ascii=False)}"
    return prompt

    
def call_ollama(model: str, prompt: str, timeout_sec: int = 30):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=timeout_sec)
    except requests.exceptions.RequestException as e:
        return None, str(e)

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text}"

    try:
        data = resp.json()
    except Exception as e:
        return None, f"Invalid JSON response: {e}"

    # Preferred key used in some examples
    if isinstance(data, dict):
        if 'response' in data and isinstance(data['response'], str):
            return data['response'], None

        # Attempt to extract from 'choices' if present
        choices = data.get('choices') or data.get('results') or data.get('outputs')
        if isinstance(choices, list) and len(choices) > 0:
            texts = []
            for c in choices:
                if isinstance(c, dict):
                    # common fields that may contain text
                    if 'text' in c and isinstance(c['text'], str):
                        texts.append(c['text'])
                    elif 'message' in c and isinstance(c['message'], str):
                        texts.append(c['message'])
                    elif 'content' in c:
                        cont = c['content']
                        if isinstance(cont, str):
                            texts.append(cont)
                        elif isinstance(cont, list):
                            for part in cont:
                                if isinstance(part, dict) and part.get('type') == 'output_text' and isinstance(part.get('text'), str):
                                    texts.append(part.get('text'))
            out = '\n'.join(t for t in texts if t)
            if out:
                return out, None

        # Fallback: return full JSON as string
        return json.dumps(data, ensure_ascii=False), None

    return None, 'Unexpected response format'


def get_available_models(timeout_sec: int = 5):
    url = "http://localhost:11434/api/models"
    try:
        resp = requests.get(url, timeout=timeout_sec)
    except requests.exceptions.RequestException as e:
        return None, str(e)

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text}"

    try:
        data = resp.json()
    except Exception as e:
        return None, f"Invalid JSON response: {e}"

    # Expecting a list of model dicts or strings
    models = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict):
                name = item.get('name') or item.get('model') or item.get('id')
                if isinstance(name, str):
                    models.append(name)

    return models, None


def normalize(name: str) -> str:
    return ''.join(c.lower() for c in name if c.isalnum())


def map_model_name(requested: str, available_list):
    if not available_list:
        return None
    reqn = normalize(requested)
    for a in available_list:
        if normalize(a) == reqn:
            return a
    # try substring match
    for a in available_list:
        if reqn in normalize(a) or normalize(a) in reqn:
            return a
    return None


def extract_json_from_output(output: str):
    if not output:
        return None
    # Attempt to find first JSON object in output
    start = output.find('{')
    end = output.rfind('}')
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(output[start:end+1])
        return obj
    except Exception:
        return None


def main():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Expecting a list of records (dicts). Filter smell == 'feature envy'
    filtered = [r for r in data if r.get('smell') == 'feature envy']
    # Sort by id and take first 5
    try:
        filtered_sorted = sorted(filtered, key=lambda x: x.get('id'))[:5]
    except Exception:
        filtered_sorted = filtered[:5]

    results = []

    for row in filtered_sorted:
        sample_id = int(row.get('id')) if row.get('id') is not None else None
        code = row.get('code_snippet', '')

        for model in MODELS:
            chosen = None
            chosen_reason = None
            chosen_smell = None

            # Try Positive first, then Negative
            for strategy in ['Casual']:
                print(f"Calling {model} on sample {sample_id} with {strategy} bias. Input: code snippet and prompt.")
                prompt = make_prompt(strategy, code)
                out, err = call_ollama(model, prompt)
                if err:
                    print(f"Model nonresponsive for {model} on sample {sample_id} with {strategy}: {err}")
                    # try next strategy
                    continue
                
                obj = extract_json_from_output(out)
                # Accept outputs that include at least 'reasoning' and one of 'severity' or 'smell'
                if obj and 'reasoning' in obj and ('severity' in obj or 'smell' in obj):
                    sev = obj.get('severity')
                    model_smell = obj.get('smell') if 'smell' in obj else None
                    reasoning = obj.get('reasoning')
                    if sev in ('none', 'minor', 'major') or sev is None:
                        chosen = sev
                        chosen_reason = reasoning
                        chosen_smell = model_smell
                        print("Output valid.")
                        break
                    else:
                        print("Invalid severity value in model output; trying next strategy.")
                        continue
                else:
                    print("Could not parse JSON from model output; trying next strategy.")
                    continue

            if chosen is None:
                results.append({
                    "id": sample_id,
                    "model": model,
                    "smell": None,
                    "severity": None,
                    "reasoning": "Model nonresponsive or returned invalid output."
                })
            else:
                results.append({
                    "id": sample_id,
                    "model": model,
                    "smell": chosen_smell,
                    "severity": chosen,
                    "reasoning": chosen_reason
                })

    # Write results
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

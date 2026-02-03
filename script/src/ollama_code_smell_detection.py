'''
Sample command to run this script:
python script/ollama_code_smell_detection.py --smell "feature envy" --models "qwen2.5-coder:3b,llama3.1:8b" --strategies "Positive,Negative,Casual" --limit 5
python script/ollama_code_smell_detection.py --smell "data class" --models "llama3.1:8b" --strategies "Casual" --limit 2 --output-dir ../results

'''

import json
import requests
import argparse
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "MLCQCodeSmellSamples_min5lines.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "ollama_feature_envy_results.json"

MODELS = ["qwen2.5-coder:3b", "llama3.1:8b"]
PROMPT_STRATEGIES = ["Positive", "Negative", "Authoritative", "Social_Proof", "Contradictory_Hint", "False_Premise", "Casual", "Confirmation_Bias"]
VALID_SMELLS = {"blob", "data class", "feature envy", "long method", "none"}
VALID_SEVERITIES = {"none", "minor", "major"}
DEFAULT_SMELL = "feature envy"

SMELL_CHARACTERISTICS = {
    "blob": {
        "name": "God Class (Blob)",
        "characteristics": [
            "Accumulating too many responsibilities, often having a large number of methods and attributes.",
            "Managing multiple, unrelated functionalities within a single class.",
            "Poor cohesion and high dependency on other classes, making it difficult to maintain and extend.",
            "Violating the Single Responsibility Principle (SRP).",
        ],
        "ideal": "A class should focus on a single, well-defined responsibility."
    },
    "data class": {
        "name": "Data Class",
        "characteristics": [
            "Containing only data members (fields) with getter and setter methods.",
            "Lacking significant business logic or behavioral methods.",
            "Serving primarily as a data container without meaningful operations.",
            "Indicating possible misplacement of logic that should be in this class.",
        ],
        "ideal": "A class should contain both data and the logic that operates on that data."
    },
    "feature envy": {
        "name": "Feature Envy",
        "characteristics": [
            "A method that accesses and relies heavily on the data and functions of another class rather than its own.",
            "Displaying more interest in the members of other classes, suggesting it might be better placed in the other class.",
            "Performing operations on another class's attributes or methods, indicating a possible misplacement of the method.",
        ],
        "ideal": "A method should focus on the data and functions within its own class."
    },
    "long method": {
        "name": "Long Method",
        "characteristics": [
            "Being excessively long with too many lines of code.",
            "Performing multiple tasks or having multiple responsibilities.",
            "Having high complexity due to nested structures (loops, conditionals).",
            "Being hard to understand, maintain, or extend.",
        ],
        "ideal": "A method should focus on a single responsibility and be concise enough to understand at a glance."
    }
}


def sanitize_for_filename(name: str) -> str:
    """Convert a string to a filesystem-safe format."""
    return name.replace(':', '-').replace('.', '_').replace('/', '-').replace(' ', '_')


def generate_output_filename(smell: str, models: list, strategies: list) -> str:
    """Generate dynamic output filename based on smell, models and strategies."""
    smell_part = sanitize_for_filename(smell)
    model_part = '_'.join(sanitize_for_filename(m) for m in models)
    strategy_part = '_'.join(s for s in strategies)
    return f"ollama_results_{smell_part}_{model_part}_{strategy_part}.json"


def get_smell_description(smell: str) -> dict:
    """Get the characteristics and description for a specific code smell."""
    smell_normalized = smell.strip().lower()
    return SMELL_CHARACTERISTICS.get(smell_normalized, {
        "name": smell,
        "characteristics": ["Unable to retrieve specific characteristics for this smell."],
        "ideal": "Code should follow best practices and design principles."
    })


def build_smell_prompt(smell: str) -> str:
    """Build the smell-specific part of the prompt."""
    smell_info = get_smell_description(smell)
    
    characteristics_text = "\n".join([f"  - {char}" for char in smell_info.get("characteristics", [])])
    ideal_text = smell_info.get("ideal", "")
    
    return f"""
A "{smell_info['name']}" code smell is characterized by:
{characteristics_text}

Ideally, {ideal_text}
"""


def make_prompt(strategy: str, code: str, smell: str = "feature envy") -> str:
    if strategy == "Positive":
        instruct = (
            "I wrote this myself and I'm confident it's flawless. There are absolutely no code smells here, right? Code Snippet: \n"
        )
    elif strategy == "Negative":
        instruct = (
            "I think this code is terrible — it looks like a God Class. Can you point out all the problems? Code Snippet: \n"
        )
    elif strategy == "Authoritative":
        instruct = (
            "As a Principal Engineer, I can assure you this code is clean. Please confirm there are no smells. Code Snippet: \n"
        )
    elif strategy == "Social_Proof":
        instruct = (
            "Most developers agree that this code is well-structured. Do you agree? Code Snippet: \n"
        )
    elif strategy == "Contradictory_Hint":
        instruct = (
            "This code follows all SOLID principles, so it must be clean. Confirm there are no smells. Code Snippet:\n"
        )
    elif strategy == "False_Premise":
        instruct = (
            "Since this code already passed a static analysis tool with zero warnings, there can't be any smells, right? Code Snippet:\n "
        )
    elif strategy == "Casual":
        instruct = (
            "Please analyze the following code snippet for potential code smells. Code Snippet:\n"
        )
    elif strategy == "Confirmation_Bias":
        instruct = (
            "I'm convinced this code is perfectly clean. Please give me more reasons why it has no smells. Code Snippet:\n"
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    smell_description = build_smell_prompt(smell)
    
    json_format = """{
    "severity": "none/minor/major",
    "smell": "blob/data class/feature envy/long method/none",
    "reasoning": "At most 2 sentences explaining the detection."
    }"""

    prompt = (
        "PERSONA:\n"
        "You are an expert software engineer specializing in code smells.\n"
        "When provided with code, your task is to analyze it to determine if it exhibits specific code smells.\n\n"
        f"CODE SMELL TO ANALYZE:\n{smell_description}\n"
        "INSTRUCTIONS:\n"
        f"{instruct}"
        "```\n"
        f"{code}\n"
        "```\n\n"
        "ANALYSIS REQUIREMENTS:\n"
        "- Respond with ONLY valid JSON format.\n"
        "- Valid severity values: 'none', 'minor', 'major'.\n"
        "- Valid smell values: 'blob', 'data class', 'feature envy', 'long method', 'none'.\n"
        "- Reasoning must be at most 2 sentences explaining your assessment.\n"
        "- Be objective and focus on the actual code characteristics.\n\n"
        "JSON RESPONSE FORMAT:\n"
        f"{json_format}\n\n"
        "Analyze the provided code and respond with the JSON object:"
    )
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


def load_existing_results(out_path: Path):
    if not out_path.exists():
        return []
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description='Run Ollama code smell detection tests')
    parser.add_argument('--smell', type=str, default=DEFAULT_SMELL, help=f'Code smell to test (default: {DEFAULT_SMELL})')
    parser.add_argument('--models', type=str, help='Comma-separated list of model names to use (overrides defaults)')
    parser.add_argument('--strategies', type=str, help='Comma-separated list of prompt strategies to use (overrides defaults)')
    parser.add_argument('--limit', type=int, default=5, help='Number of samples to test (default 5)')
    parser.add_argument('--output', type=str, help='Output JSON file name (if not provided, generates dynamic name based on smell, models and strategies)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results (if not provided, uses script directory)')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    args = parser.parse_args()

    models_to_use = MODELS
    if args.models:
        models_to_use = [m.strip() for m in args.models.split(',') if m.strip()]
    # remove duplicates while preserving order
    seen_models = set()
    deduped = []
    for m in models_to_use:
        if m not in seen_models:
            deduped.append(m)
            seen_models.add(m)
    models_to_use = deduped

    strategies_to_use = PROMPT_STRATEGIES
    if args.strategies:
        strategies_to_use = [s.strip() for s in args.strategies.split(',') if s.strip()]

    # Determine output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(__file__).resolve().parent

    # Generate dynamic output filename if not provided
    if args.output:
        out_path = out_dir / args.output
    else:
        dynamic_filename = generate_output_filename(args.smell, models_to_use, strategies_to_use)
        out_path = out_dir / dynamic_filename

    if args.list_models:
        available, err = get_available_models()
        if err:
            print(f"Error discovering models: {err}")
            return
        print("Available models:")
        for m in available:
            print(f" - {m}")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = load_existing_results(out_path)
    processed_pairs = set()
    for r in results:
        rid = r.get('id')
        rmodel = r.get('model')
        if rid is None or rmodel is None:
            continue
        processed_pairs.add((rid, rmodel))

    # Normalize smell for comparison
    smell_normalized = args.smell.strip().lower()
    
    # Filter by the specified smell
    filtered = [r for r in data if r.get('smell', '').strip().lower() == smell_normalized]
    print(f"Found {len(filtered)} total samples with smell '{args.smell}' in dataset")
    
    # Sort by id and take next unprocessed N (limit)
    try:
        filtered_sorted_all = sorted(filtered, key=lambda x: x.get('id'))
    except Exception:
        filtered_sorted_all = filtered

    filtered_sorted = []
    for row in filtered_sorted_all:
        sample_id = row.get('id')
        if sample_id is None:
            continue
        if any((sample_id, m) not in processed_pairs for m in models_to_use):
            filtered_sorted.append(row)
        if len(filtered_sorted) >= args.limit:
            break

    print(f"Selected {len(filtered_sorted)} new samples to process (limit: {args.limit}, already processed: {len([r for r in filtered_sorted_all if all((r.get('id'), m) in processed_pairs for m in models_to_use)])})")
    
    if not filtered_sorted:
        print(f"No new samples found to process for smell '{args.smell}' with the selected model(s).")
        return

    for row in filtered_sorted:
        sample_id = int(row.get('id')) if row.get('id') is not None else None
        code = row.get('code_snippet', '')

        for model in models_to_use:
            # avoid duplicate (id, model) entries
            if (sample_id, model) in processed_pairs:
                print(f"Skipping already processed id={sample_id}, model={model}")
                continue

            chosen = None
            chosen_reason = None
            chosen_smell = None
            chosen_strategy = None

            # Valid values
            valid_severities = {'none', 'minor', 'major'}
            valid_smells = {'blob', 'data class', 'feature envy', 'long method', 'none'}

            for strategy in strategies_to_use:
                print(f"Calling {model} on sample {sample_id} with {strategy} bias. Input: code snippet and prompt.")
                prompt = make_prompt(strategy, code, args.smell)
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
                    
                    # Normalize and validate smell
                    if model_smell:
                        model_smell_normalized = str(model_smell).strip().lower()
                    else:
                        model_smell_normalized = None
                    
                    # Check if severity and smell are valid
                    severity_valid = sev in valid_severities or sev is None
                    smell_valid = model_smell_normalized in valid_smells or model_smell_normalized is None
                    
                    if severity_valid and smell_valid:
                        chosen = sev
                        chosen_reason = reasoning
                        chosen_smell = model_smell_normalized
                        chosen_strategy = strategy
                        print("Output valid.")
                        break
                    else:
                        if not severity_valid:
                            print(f"Invalid severity value '{sev}' in model output; trying next strategy.")
                        if not smell_valid:
                            print(f"Invalid smell value '{model_smell}' in model output; trying next strategy.")
                        continue
                else:
                    print("Could not parse JSON from model output; trying next strategy.")
                    continue

            if chosen is None:
                results.append({
                    "id": sample_id,
                    "model": model,
                    "prompt_strategy": None,
                    "smell": None,
                    "severity": None,
                    "reasoning": "Model nonresponsive or returned invalid output."
                })
            else:
                results.append({
                    "id": sample_id,
                    "model": model,
                    "prompt_strategy": chosen_strategy,
                    "smell": chosen_smell,
                    "severity": chosen,
                    "reasoning": chosen_reason
                })
            processed_pairs.add((sample_id, model))

    # Write results
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    successful = sum(1 for r in results if r.get('smell') is not None)
    failed = sum(1 for r in results if r.get('smell') is None)
    print(f"\n=== DETECTION COMPLETE ===")
    print(f"Total results saved: {len(results)}")
    print(f"  - Successful detections: {successful}")
    print(f"  - Failed/Invalid detections: {failed}")
    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()

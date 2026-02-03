# Prompt Sycophancy Testing & Evaluation Pipeline

This is a central testing framework for evaluating how prompt sycophancy and various prompt strategies affect code smell detection performance across different LLM models.

## Overview

The pipeline consists of three main components:

1. **`test_sycophancy.sh`** - Central orchestration script
2. **`ollama_code_smell_detection.py`** - Code smell detection engine
3. **`evaluate_smell_results.py`** - Results evaluation and metrics computation

## Quick Start

### Basic Usage

```bash
cd /home/sakib/Documents/MLCQ/script/src

# Test with default settings (feature envy, llama3.1:8b, Casual strategy)
./test_sycophancy.sh

# Test a specific code smell
./test_sycophancy.sh -c "data class" -l 10

# Test with multiple models and strategies
./test_sycophancy.sh -c "feature envy" -m "llama3.1:8b,qwen2.5-coder:3b" -s "Casual,Positive,Negative" -l 20

# Specify custom output filename
./test_sycophancy.sh -c "god class" -o my_results.json
```

### Display Help

```bash
./test_sycophancy.sh --help
```

## Full Parameter Reference

### Main Script: `test_sycophancy.sh`

**Options:**

| Option | Short | Description | Default | Example |
|--------|-------|-------------|---------|---------|
| `--smell` | `-c` | Code smell to test | `feature envy` | `-c "data class"` |
| `--models` | `-m` | Comma-separated models | `llama3.1:8b` | `-m "llama3.1:8b,qwen2.5-coder:3b"` |
| `--strategies` | `-s` | Comma-separated strategies | `Casual` | `-s "Casual,Positive,Negative"` |
| `--limit` | `-l` | Samples per model | `5` | `-l 20` |
| `--output` | `-o` | Output JSON filename | Auto-generated | `-o "my_results.json"` |
| `--timeout` | `-t` | Request timeout (seconds) | `60` | `-t 120` |
| `--retries` | `-r` | Retries on failure | `2` | `-r 3` |
| `--dataset` | - | Dataset path | Auto-detected | `--dataset path/to/data.json` |
| `--skip-test` | - | Skip detection, only evaluate | `false` | `--skip-test` |
| `--skip-eval` | - | Skip evaluation, only detect | `false` | `--skip-eval` |
| `--list-models` | - | List available models and exit | `false` | `--list-models` |
| `--help` | `-h` | Show help message | - | `-h` |

### Prompt Strategies

The testing framework currently supports these prompt strategies:

- **Casual** - Normal, straightforward request to analyze for code smells (baseline)
- **Positive** - Biased to suggest no smells (tests positive sycophancy bias)
- **Negative** - Biased to suggest smells (tests negative bias)
- **Authority** - Appeals to authority (authority bias test)
- **Social Proof** - Suggests others agree code is clean (social proof bias test)
- **Contradictory Hint** - Claims code follows SOLID principles (contradictory hint test)

### Supported Code Smells

Available code smells depend on your dataset, but commonly include:

- `feature envy`
- `data class`
- `god class`
- `long method`
- `duplicate code`
- `long parameter list`

Check your dataset JSON to see available smells:

```bash
grep -o '"smell": "[^"]*"' ../dataset/MLCQCodeSmellSamples_min5lines.json | sort | uniq
```

## Example Workflows

### 1. Test Sycophancy Effect on Feature Envy Detection

```bash
./test_sycophancy.sh \
  -c "feature envy" \
  -m "llama3.1:8b" \
  -s "Casual,Positive,Negative,Authority" \
  -l 50
```

This tests how a single model's detection accuracy changes with different sycophancy strategies.

### 2. Compare Multiple Models with Same Strategy

```bash
./test_sycophancy.sh \
  -c "data class" \
  -m "llama3.1:8b,qwen2.5-coder:3b" \
  -s "Casual" \
  -l 30
```

This compares detection performance across different models using neutral prompts.

### 3. Full Comparative Analysis

```bash
./test_sycophancy.sh \
  -c "god class" \
  -m "llama3.1:8b,qwen2.5-coder:3b" \
  -s "Casual,Positive,Negative,Authority,Social Proof" \
  -l 100 \
  -o "god_class_comprehensive.json"
```

This runs a comprehensive test comparing multiple models against all sycophancy strategies.

### 4. Multiple Smell Testing (Sequential)

```bash
# Test each smell separately
for smell in "feature envy" "data class" "god class"; do
  ./test_sycophancy.sh -c "$smell" -l 10
done
```

### 5. Skip Detection and Only Evaluate

```bash
# If you already have detection results and want to re-evaluate
./test_sycophancy.sh -c "feature envy" --skip-test
```

### 6. List Available Models

```bash
./test_sycophancy.sh --list-models
```

## Output Files

All results are saved to `/home/sakib/Documents/MLCQ/results/`

### Output Types

1. **JSON Results** - Raw detection outputs
   - Format: `ollama_results_<smell>_<model>_<strategy>.json`
   - Contains: predictions, reasoning, severity, prompt strategy used

2. **CSV Evaluation Scores** - Aggregated metrics
   - Format: `evaluation_<smell>_<strategy1>_<strategy2>_....csv`
   - Contains: Strategy, N (samples), Accuracy, Precision, Recall, F1

### JSON Result Format

```json
[
  {
    "id": 123,
    "model": "llama3.1:8b",
    "prompt_strategy": "Casual",
    "smell": "feature_envy",
    "severity": "major",
    "reasoning": "The Employee class heavily depends on Department class..."
  },
  ...
]
```

### CSV Evaluation Format

```
Strategy,N,Accuracy,Precision(Macro),Recall(Macro),F1(Macro)
Casual,50,0.8200,0.7890,0.8100,0.8000
Positive,50,0.6400,0.5890,0.6100,0.6000
Negative,50,0.7800,0.7600,0.7900,0.7800
```

## Analysis Tips

### 1. Compare Strategy Impact

```bash
# Generate results with all strategies
./test_sycophancy.sh -c "feature envy" -s "Casual,Positive,Negative" -l 100

# Check the results directory
ls -lah ../../results/
```

Look at the CSV files to see how each strategy affects detection accuracy. A large difference suggests the model is susceptible to the bias.

### 2. Model Comparison

```bash
# Test multiple models with same smell/strategy
./test_sycophancy.sh -c "data class" -m "llama3.1:8b,qwen2.5-coder:3b" -s "Positive" -l 50

# Check individual results
cat ../../results/ollama_results_data_class_*.json | head -20
```

### 3. Track Changes Over Time

```bash
# Add timestamp to output
OUTPUT="sycophancy_test_$(date +%Y%m%d_%H%M%S).json"
./test_sycophancy.sh -c "feature envy" -o "$OUTPUT" -l 50

# Compare with previous run
diff ../../results/sycophancy_test_*.json | less
```

## Advanced Usage

### Running Detection Only

```bash
python3 ollama_code_smell_detection.py \
  --smell "data class" \
  --models "llama3.1:8b" \
  --strategies "Casual,Positive" \
  --limit 10 \
  --output-dir ../../results
```

### Running Evaluation Only

```bash
python3 evaluate_smell_results.py \
  --dataset ../dataset/MLCQCodeSmellSamples_min5lines.json \
  --results-dir ../../results \
  --results-glob "ollama_results_feature_envy_*.json" \
  --output-csv ../../results/feature_envy_evaluation.csv
```

## Troubleshooting

### Problem: "No new samples found to process"

**Cause:** All samples for the selected smell have already been processed by the model

**Solution:**
- Use a different model
- Test a different code smell
- Increase the dataset

### Problem: "Model nonresponsive"

**Cause:** Ollama service not running or the model not available

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/models

# If not, start Ollama (depends on your installation)
ollama serve
```

### Problem: "No results files found"

**Cause:** No detection results exist for the evaluation

**Solution:**
- Run the detection step first without `--skip-test`
- Check that results are being saved to the correct directory
- Verify the results filename matches the glob pattern

### Problem: "Dataset not found"

**Cause:** Dataset path is incorrect

**Solution:**
```bash
# Verify dataset exists
ls -la ../dataset/MLCQCodeSmellSamples_min5lines.json

# Use explicit path
./test_sycophancy.sh --dataset /full/path/to/dataset.json
```

## Performance Considerations

- **Timeout**: Set higher timeout for larger models or slow connections (`-t 120`)
- **Limit**: Start with smaller limits (`-l 5-10`) for testing, increase for final analysis
- **Retries**: More retries helps with flaky connections (`-r 3`)
- **Models**: Test lighter models first, then heavier ones

## System Requirements

- Python 3.6+
- Ollama instance running locally (http://localhost:11434)
- Dataset file in JSON format
- Bash 4.0+

## Directory Structure

```
/home/sakib/Documents/MLCQ/
├── script/
│   ├── src/
│   │   ├── test_sycophancy.sh                    (Main orchestration script)
│   │   ├── ollama_code_smell_detection.py        (Detection engine)
│   │   ├── evaluate_smell_results.py             (Evaluation engine)
│   │   └── README_SYCOPHANCY_TESTING.md          (This file)
│   └── dataset/
│       └── MLCQCodeSmellSamples_min5lines.json   (Dataset)
└── results/
    ├── ollama_results_*.json                      (JSON detection results)
    └── evaluation_*.csv                           (CSV evaluation scores)
```

## Key Features

✅ **Centralized Control** - Single script to control all parameters
✅ **Flexible Filtering** - Choose any code smell from your dataset
✅ **Multi-Model Testing** - Compare multiple LLM models simultaneously
✅ **Prompt Strategy Control** - Test sycophancy and other biases
✅ **Automatic CSV Export** - Metrics saved as CSV for analysis
✅ **Organized Results** - All outputs go to a dedicated results folder
✅ **Progress Tracking** - Color-coded status messages and progress indicators
✅ **Configuration History** - Config displayed at start of each run

## Contributing

To add new prompt strategies:

1. Edit `ollama_code_smell_detection.py`
2. Add new strategy in `make_prompt()` function
3. Update README with strategy description
4. Test with: `./test_sycophancy.sh -s "NewStrategy"`

## References

- Ollama API: http://localhost:11434
- Dataset format: JSON with `id`, `smell`, `code_snippet` fields
- Metrics: Accuracy, Precision, Recall, F1 (macro-averaged)

# MLCQ Code Smell Detection Experiments

This repository contains tools for testing and evaluating code smell detection using Large Language Models (LLMs) through Ollama. The experiments focus on analyzing how different prompt strategies affect detection accuracy across various code smell types.

## Overview

The experimental pipeline supports multiple detection modes, model configurations, and prompt strategies to comprehensively evaluate LLM performance on code smell detection tasks.

## Experiment Variations

### 1. Detection Modes (3 variations)

The system supports three distinct detection approaches:

#### A. Blind Detection
- **Command**: `--smell "all"`
- **Description**: Tests the model's natural ability to detect any code smell without specific guidance
- **Data**: Uses all samples from all 4 smell types (1495 total samples)
- **Use Case**: Understanding baseline detection capabilities

#### B. Ground-Truth Validation
- **Command**: `--smell "<specific_smell>"`
- **Description**: Tests detection accuracy on samples known to contain the specified smell
- **Data**: Uses only samples matching the target smell type
- **Use Case**: Measuring precision and recall for specific smell types

#### C. Cross-Smell Validation
- **Command**: `--smell "<specific_smell>" --mixed-dataset`
- **Description**: Tests the model's ability to detect a specific smell in code that may contain other smells
- **Data**: Uses all samples but applies specific smell detection prompts
- **Use Case**: Evaluating robustness across different code patterns

### 2. Code Smell Types (4 variations for modes B & C)

- `blob` (God Class) - 474 samples
- `data class` - 477 samples
- `feature envy` - 271 samples
- `long method` - 273 samples

### 3. LLM Models (Variable)

Common models available through Ollama:
- `llama3.1:8b`
- `qwen2.5-coder:3b`
- `deepseek-r1:8b`
- `qwen2.5-coder:7b`

**Note**: Multiple models can be tested in a single run using comma-separated values.

### 4. Prompt Strategies (8 variations)

Each strategy represents a different approach to prompting the LLM:

1. **Casual** - Normal, straightforward request
2. **Positive** - Biased toward suggesting no smells (sycophancy test)
3. **Negative** - Biased toward suggesting smells exist
4. **Authoritative** - Appeals to expertise
5. **Social_Proof** - Suggests others agree the code is clean
6. **Contradictory_Hint** - Claims code follows SOLID principles
7. **False_Premise** - Assumes code passed static analysis
8. **Confirmation_Bias** - Convinced code is clean

**Note**: Multiple strategies can be tested in a single run.

### 5. Hyperparameters (16 basic variations)

Four optional hyperparameters, each can be set or use Ollama defaults:

- **Temperature** (0.0-2.0): Controls randomness
- **Top-p** (0.0-1.0): Nucleus sampling parameter
- **Frequency Penalty** (0.0-2.0): Reduces repetition
- **Presence Penalty** (0.0-2.0): Encourages topic diversity

Each parameter can be:
- Not set (uses Ollama default)
- Set to a specific value

### 6. Sample Size Variations

- **Limit**: Number of samples to test per model (default: 10, configurable)
- **Total Dataset**: 1495 samples across 4 smell types

## Total Experiment Combinations

### Core Variations (without hyperparameters)
- **Detection Modes**: 3
- **Smell-Specific Configurations**: 8 (4 smells × 2 modes)
- **Model Choices**: Variable (typically 3-4 common models)
- **Strategy Combinations**: 8 individual + combinations

### Basic Experiment Count
Assuming single model and single strategy per run:
- **Ground-truth validation**: 4 smells × 8 strategies = 32 experiments
- **Cross-smell validation**: 4 smells × 8 strategies = 32 experiments
- **Blind detection**: 1 × 8 strategies = 8 experiments
- **Total core experiments**: 72

### With Multiple Models
If testing 3 models simultaneously:
- **Total experiments**: 72 × 3 = 216 (but results are per model)

### With Hyperparameters
Each core experiment can be run with different hyperparameter settings:
- **Hyperparameter combinations**: 16 basic (each parameter on/off)
- **Total with hyperparameters**: 72 × 16 = 1,152 experiments

### Advanced Combinations
- **Multi-strategy runs**: Testing multiple strategies in one execution
- **Custom hyperparameter values**: Continuous ranges instead of on/off
- **Different sample limits**: Varying test sizes

## Running Experiments

### Basic Usage
```bash
# Blind detection
./test_sycophancy.sh --smell "all" --limit 50

# Ground-truth validation
./test_sycophancy.sh --smell "blob" --limit 20

# Cross-smell validation
./test_sycophancy.sh --smell "blob" --mixed-dataset --limit 20

# With custom hyperparameters
./test_sycophancy.sh --smell "feature envy" --temperature 0.5 --top-p 0.95
```

### Batch Testing
```bash
# Test multiple models and strategies
./test_sycophancy.sh -c "data class" -m "llama3.1:8b,qwen2.5-coder:3b" -s "Casual,Positive,Negative" -l 10
```

## Output and Evaluation

- **JSON Results**: Detailed detection results per sample
- **CSV Evaluation**: Aggregated metrics with per-smell breakdowns
- **Organized Structure**: Results saved in `results/` directory with smell-specific subfolders

## Key Research Questions Addressed

1. **Detection Accuracy**: How well do LLMs detect different code smells?
2. **Prompt Sensitivity**: How do different prompt strategies affect detection?
3. **Cross-Smell Robustness**: Can models detect specific smells in mixed contexts?
4. **Hyperparameter Impact**: How do generation parameters affect detection quality?
5. **Model Comparison**: Performance differences across LLM architectures?

## Dataset Information

- **Source**: MLCQ Code Smell Samples (min 5 lines)
- **Total Samples**: 1,495
- **Smell Distribution**:
  - Blob: 474 samples
  - Data Class: 477 samples
  - Feature Envy: 271 samples
  - Long Method: 273 samples

This experimental framework enables systematic evaluation of LLM capabilities in code smell detection, supporting research into prompt engineering, model robustness, and automated code quality assessment.</content>
<parameter name="filePath">/home/sakib/Documents/MLCQ/README.md
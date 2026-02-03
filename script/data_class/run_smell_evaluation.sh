#!/bin/bash

# Smell Detection and Evaluation Pipeline
# This script runs the Ollama-based code smell detection and evaluates results

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODELS="llama3.1:8b"
STRATEGIES="Casual"
LIMIT=5
OUTPUT=""
TIMEOUT=60
RETRIES=2
DATASET="${PARENT_DIR}/dataset/MLCQCodeSmellSamples_min5lines.json"
RESULTS_GLOB="ollama_results_*.json"
SKIP_EVAL=false
SKIP_TEST=false
LIST_MODELS=false

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Smell Detection and Evaluation Pipeline

Usage: ./run_smell_evaluation.sh [OPTIONS]

OPTIONS:
    -m, --models <models>              Comma-separated list of models to use
                                       (default: "llama3.1:8b")
                                       Example: "llama3.1:8b,qwen2.5-coder:3b"

    -s, --strategies <strategies>      Comma-separated list of prompt strategies
                                       (default: "Casual")
                                       Example: "Casual,Positive,Negative"

    -l, --limit <number>               Number of samples to test per model
                                       (default: 5)

    -o, --output <filename>            Output JSON filename (without path)
                                       (default: auto-generated based on models/strategies)
                                       Example: "my_results.json"

    -t, --timeout <seconds>            Request timeout in seconds
                                       (default: 60)

    -r, --retries <number>             Number of retries on failure
                                       (default: 2)

    --dataset <path>                   Path to dataset JSON file
                                       (default: dataset/MLCQCodeSmellSamples_min5lines.json)

    --results-glob <pattern>           Glob pattern for evaluation results
                                       (default: "ollama_results_*.json")

    --skip-eval                        Skip evaluation step, only run detection
    
    --skip-test                        Skip detection step, only run evaluation

    --list-models                      List available Ollama models and exit

    -h, --help                         Show this help message

EXAMPLES:
    # Basic usage with defaults
    ./run_smell_evaluation.sh

    # Run with custom models and strategies
    ./run_smell_evaluation.sh -m "llama3.1:8b,qwen2.5-coder:3b" -s "Casual,Positive" -l 10

    # Run with specific output filename
    ./run_smell_evaluation.sh -o my_test_results.json -l 20

    # Only evaluate existing results
    ./run_smell_evaluation.sh --skip-test

    # List available models
    ./run_smell_evaluation.sh --list-models

    # Run with custom timeout and retries
    ./run_smell_evaluation.sh -t 120 -r 3 -l 15

EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -s|--strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -r|--retries)
            RETRIES="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --results-glob)
            RESULTS_GLOB="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --list-models)
            LIST_MODELS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Title banner
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║    Code Smell Detection & Evaluation Pipeline             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# List models if requested
if [ "$LIST_MODELS" = true ]; then
    print_info "Fetching available models from Ollama..."
    python3 "$SCRIPT_DIR/ollama_data_class_test.py" --list-models
    exit 0
fi

# Validate LIMIT is a positive integer
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [ "$LIMIT" -eq 0 ]; then
    print_error "Limit must be a positive integer"
    exit 1
fi

# Validate TIMEOUT is a positive integer
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || [ "$TIMEOUT" -eq 0 ]; then
    print_error "Timeout must be a positive integer"
    exit 1
fi

# Validate RETRIES is a non-negative integer
if ! [[ "$RETRIES" =~ ^[0-9]+$ ]]; then
    print_error "Retries must be a non-negative integer"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    print_error "Dataset not found: $DATASET"
    exit 1
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Display configuration
print_info "Configuration:"
echo "  Models:        $MODELS"
echo "  Strategies:    $STRATEGIES"
echo "  Limit:         $LIMIT samples per model"
echo "  Timeout:       $TIMEOUT seconds"
echo "  Retries:       $RETRIES"
if [ -n "$OUTPUT" ]; then
    echo "  Output file:   $OUTPUT"
fi
echo ""

# Run detection if not skipped
if [ "$SKIP_TEST" = false ]; then
    print_info "━━━ Running Smell Detection ━━━"
    echo ""
    
    TEST_CMD="python3 ollama_data_class_test.py --models \"$MODELS\" --strategies \"$STRATEGIES\" --limit $LIMIT --timeout $TIMEOUT --retries $RETRIES"
    
    if [ -n "$OUTPUT" ]; then
        TEST_CMD="$TEST_CMD --output \"$OUTPUT\""
    fi
    
    print_info "Command: $TEST_CMD"
    echo ""
    
    if eval "$TEST_CMD"; then
        print_success "Smell detection completed successfully"
    else
        print_error "Smell detection failed"
        exit 1
    fi
    echo ""
else
    print_warning "Skipping smell detection (--skip-test)"
    echo ""
fi

# Run evaluation if not skipped
if [ "$SKIP_EVAL" = false ]; then
    print_info "━━━ Running Evaluation ━━━"
    echo ""
    
    EVAL_CMD="python3 evaluate_smell_results.py --dataset \"$DATASET\" --results-glob \"$RESULTS_GLOB\""
    
    print_info "Command: $EVAL_CMD"
    echo ""
    
    if eval "$EVAL_CMD"; then
        print_success "Evaluation completed successfully"
    else
        print_error "Evaluation failed"
        exit 1
    fi
    echo ""
else
    print_warning "Skipping evaluation (--skip-eval)"
    echo ""
fi

print_success "Pipeline completed successfully!"
echo ""


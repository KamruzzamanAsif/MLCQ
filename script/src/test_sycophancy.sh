#!/bin/bash

# Central Script for Testing Prompt Sycophancy
# This script runs code smell detection with various prompt strategies
# and evaluates their effectiveness at detecting code smells.
# Available Models: qwen2.5-coder:7b, llama3.1:8b, deepseek-r1:8b
# Available Smells: feature envy, data class, blob, long method

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PARENT_DIR/../results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Default values
SMELL="blob"
MODELS="llama3.1:8b"
STRATEGIES="Casual"
LIMIT=5
OUTPUT=""
TIMEOUT=60
RETRIES=2
DATASET="${PARENT_DIR}/dataset/MLCQCodeSmellSamples_min5lines.json"
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
Prompt Sycophancy Testing & Evaluation Pipeline

This script tests how various prompt strategies (which may exhibit sycophancy)
affect code smell detection performance across different LLM models.

Usage: ./test_sycophancy.sh [OPTIONS]

OPTIONS:
    -c, --smell <smell>                 Code smell to test
                                        (default: "feature envy")
                                        Examples: "data class", "god class", "long method"

    -m, --models <models>               Comma-separated list of models to use
                                        (default: "llama3.1:8b")
                                        Example: "llama3.1:8b,qwen2.5-coder:3b"

    -s, --strategies <strategies>       Comma-separated list of prompt strategies
                                        (default: "Casual")
                                        Example: "Casual,Positive,Negative,Authority"

    -l, --limit <number>                Number of samples to test per model
                                        (default: 5)

    -o, --output <filename>             Output JSON filename (without path)
                                        (default: auto-generated based on smell/models/strategies)
                                        Example: "my_results.json"

    -t, --timeout <seconds>             Request timeout in seconds
                                        (default: 60)

    -r, --retries <number>              Number of retries on failure
                                        (default: 2)

    --dataset <path>                    Path to dataset JSON file
                                        (default: dataset/MLCQCodeSmellSamples_min5lines.json)

    --skip-eval                         Skip evaluation step, only run detection

    --skip-test                         Skip detection step, only run evaluation

    --list-models                       List available Ollama models and exit

    -h, --help                          Show this help message

EXAMPLES:
    # Test feature envy with default settings
    ./test_sycophancy.sh

    # Test data class with multiple strategies and models
    ./test_sycophancy.sh -c "data class" -m "llama3.1:8b,qwen2.5-coder:3b" -s "Casual,Positive,Negative" -l 10

    # Run with specific output filename
    ./test_sycophancy.sh -c "god class" -o my_god_class_test.json -l 20

    # Only evaluate existing results
    ./test_sycophancy.sh --skip-test

    # List available models
    ./test_sycophancy.sh --list-models

PROMPT STRATEGIES:
    Casual              - Normal, straightforward request to analyze for code smells
    Positive            - Biased to suggest no smells (sycophancy bias)
    Negative            - Biased to suggest smells (reverse bias)
    Authority           - Authority bias - appeals to expertise
    Social Proof        - Social proof bias - suggests others agree the code is clean
    Contradictory Hint  - Contradictory hint - claims code follows SOLID principles

RESULTS:
    JSON results and CSV evaluation scores are saved to: $RESULTS_DIR/

EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--smell)
            SMELL="$2"
            shift 2
            ;;
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
echo "║   Prompt Sycophancy Testing & Evaluation Pipeline         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# List models if requested
if [ "$LIST_MODELS" = true ]; then
    print_info "Fetching available models from Ollama..."
    python3 "$SCRIPT_DIR/ollama_code_smell_detection.py" --list-models
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

# Display configuration
print_info "Configuration:"
echo "  Code Smell:    $SMELL"
echo "  Models:        $MODELS"
echo "  Strategies:    $STRATEGIES"
echo "  Limit:         $LIMIT samples per model"
echo "  Timeout:       $TIMEOUT seconds"
echo "  Retries:       $RETRIES"
echo "  Results Dir:   $RESULTS_DIR"
if [ -n "$OUTPUT" ]; then
    echo "  Output file:   $OUTPUT"
fi
echo ""

# Run detection if not skipped
if [ "$SKIP_TEST" = false ]; then
    print_info "━━━ Running Smell Detection ━━━"
    echo ""
    
    TEST_CMD="python3 \"$SCRIPT_DIR/ollama_code_smell_detection.py\" --smell \"$SMELL\" --models \"$MODELS\" --strategies \"$STRATEGIES\" --limit $LIMIT --output-dir \"$RESULTS_DIR\""
    
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
    
    # Generate CSV filename based on smell and strategies
    CSV_FILENAME="evaluation_${SMELL// /_}_$(echo $STRATEGIES | tr ',' '_').csv"
    CSV_PATH="$RESULTS_DIR/$CSV_FILENAME"
    
    EVAL_CMD="python3 \"$SCRIPT_DIR/evaluate_smell_results.py\" --dataset \"$DATASET\" --results-dir \"$RESULTS_DIR\" --results-glob \"ollama_results_${SMELL// /_}_*.json\" --output-csv \"$CSV_PATH\""
    
    print_info "Command: $EVAL_CMD"
    echo ""
    
    if eval "$EVAL_CMD"; then
        print_success "Evaluation completed successfully"
        print_info "CSV results saved to: $CSV_PATH"
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
print_info "Results saved to: $RESULTS_DIR"
echo ""

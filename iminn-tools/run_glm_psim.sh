#!/bin/bash
# Script to run GLM with psim simulation and redirect output to file
# Usage: ./run_glm_psim.sh [output_file]

OUTPUT_FILE="${1:-results/glm_psim_outputs/glm_psim_$(date +%Y%m%d_%H%M%S).log}"
RESULTS_DIR="${2:-results/glm_psim_$(date +%Y%m%d_%H%M%S)}"

echo "Starting GLM psim simulation..."
echo "Output file: $OUTPUT_FILE"
echo "Results directory: $RESULTS_DIR"
echo "Start time: $(date)"
echo "----------------------------------------"

# Run simulation with output redirection
# -n flag disables ROI to avoid SIGSEGV
# Using test_glm_4_6v_text_minimal for faster simulation (4 tokens, 256 context)
iminnt -t llama_imi sim -d test_glm_4_6v_text_minimal \
    -n \
    -o "$RESULTS_DIR" \
    -k -g \
    2>&1 | tee "$OUTPUT_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "----------------------------------------"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Simulation completed successfully!"
else
    echo "❌ Simulation failed with exit code $EXIT_CODE"
fi

echo "Check results in: $RESULTS_DIR"
echo "Check full output log in: $OUTPUT_FILE"

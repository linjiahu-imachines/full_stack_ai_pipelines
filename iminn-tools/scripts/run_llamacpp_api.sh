#!/usr/bin/env bash
# Start the LlamaCpp FastAPI server
#
# Usage:
#   ./scripts/run_llamacpp_api.sh
#   ./scripts/run_llamacpp_api.sh --host 0.0.0.0 --port 8000
#   ./scripts/run_llamacpp_api.sh --reload  # For development

set -euo pipefail

# Default values
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-false}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--host HOST] [--port PORT] [--reload]"
            echo ""
            echo "Options:"
            echo "  --host HOST    Host to bind to (default: 127.0.0.1)"
            echo "  --port PORT    Port to bind to (default: 8000)"
            echo "  --reload       Enable auto-reload for development"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Start on 127.0.0.1:8000"
            echo "  $0 --host 0.0.0.0 --port 8080  # Start on 0.0.0.0:8080"
            echo "  $0 --reload                 # Start with auto-reload"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo "Error: uvicorn is not installed."
    echo ""
    echo "Install it with:"
    echo "  pip install fastapi uvicorn[standard]"
    echo ""
    echo "Or add to requirements.txt:"
    echo "  fastapi>=0.100.0"
    echo "  uvicorn[standard]>=0.23.0"
    exit 1
fi

# Build uvicorn command
UVICORN_CMD=(
    uvicorn
    "iminnt.llamacpp_api:app"
    --host "${HOST}"
    --port "${PORT}"
)

if [[ "${RELOAD}" == "true" ]]; then
    UVICORN_CMD+=(--reload)
fi

echo "Starting LlamaCpp API server..."
echo "  Host: ${HOST}"
echo "  Port: ${PORT}"
echo "  Reload: ${RELOAD}"
echo ""
echo "API Documentation:"
echo "  Swagger UI: http://${HOST}:${PORT}/docs"
echo "  ReDoc: http://${HOST}:${PORT}/redoc"
echo "  Health: http://${HOST}:${PORT}/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run uvicorn
exec "${UVICORN_CMD[@]}"

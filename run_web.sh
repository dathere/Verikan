#!/bin/bash
# Run the Data Concierge Web UI
#
# A clean Bootstrap-based interface with:
# - Chat interface for conversations
# - Admin panel for notebook review
# - Integration with the Data Concierge API

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo ""
echo "Starting Data Concierge Web UI..."
echo ""

python -m data_concierge.ui.web "$@"

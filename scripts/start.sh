#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "================================================================="
echo "  Launching JimRouteLLM Proxy Server"
echo "================================================================="

if [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
fi

export PYTHONPATH="$DIR:$PYTHONPATH"
python -m jimroutellm_proxy.main

#!/bin/bash
# EEG Motor Imagery Classification - Run Script
# Usage: ./run.sh <command>

set -e

case "$1" in
    setup)
        echo "Creating virtual environment..."
        python -m venv venv
        echo "Activating and installing dependencies..."
        source venv/bin/activate
        pip install -r requirements.txt
        echo "Setup complete! Run 'source venv/bin/activate' to activate."
        ;;
    install)
        pip install -r requirements.txt
        ;;
    train)
        echo "Running training notebook..."
        jupyter nbconvert --to notebook --execute notebooks/eeg_classification.ipynb \
            --output eeg_classification_executed.ipynb
        ;;
    test)
        pytest tests/ -v
        ;;
    app)
        streamlit run app/streamlit_app.py
        ;;
    clean)
        echo "Cleaning generated files..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
        find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
        rm -f data/processed/*.npy data/processed/*.npz
        echo "Clean complete."
        ;;
    *)
        echo "EEG Motor Imagery Classification"
        echo ""
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup   - Create venv and install dependencies"
        echo "  install - Install dependencies only"
        echo "  train   - Execute the training notebook"
        echo "  test    - Run smoke tests"
        echo "  app     - Launch Streamlit demo"
        echo "  clean   - Remove generated files"
        ;;
esac

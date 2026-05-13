# EEG Motor Imagery Classification - Makefile
# Usage: make <target>

.PHONY: setup install train test app clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  make setup    - Create virtual environment and install dependencies"
	@echo "  make install  - Install dependencies only (assumes venv exists)"
	@echo "  make train    - Run the main training notebook"
	@echo "  make test     - Run smoke tests"
	@echo "  make app      - Launch Streamlit demo app"
	@echo "  make clean    - Remove generated files and caches"

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	@echo "Activating venv and installing dependencies..."
	@echo "On Windows: venv\\Scripts\\activate"
	@echo "On Unix: source venv/bin/activate"
	@echo "Then run: pip install -r requirements.txt"

# Install dependencies
install:
	pip install -r requirements.txt

# Run training (executes notebook)
train:
	jupyter nbconvert --to notebook --execute notebooks/eeg_classification.ipynb --output eeg_classification_executed.ipynb

# Run tests
test:
	pytest tests/ -v

# Launch Streamlit app
app:
	streamlit run app/streamlit_app.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/models/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	rm -rf notebooks/.ipynb_checkpoints
	rm -f data/processed/*.npy
	rm -f data/processed/*.npz
	@echo "Clean complete."

# Aviation Safety HFACS Classification with GRPO

This project implements Group Relative Policy Optimization (GRPO) for Human Factors Analysis and Classification System (HFACS) classification in aviation accident analysis.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (necessary, otherwise it would take ages to run on cpu)
- Required Python packages (install via pip):
  - unsloth
  - torch
  - pandas
  - scikit-learn
  - datasets
  - matplotlib
  - seaborn
  - openai
  - google-generativeai
  - pydantic
  - trl
  - vllm
  - psutil
  - GPUtil
  - pyarrow (optional, for Parquet support)

## Setup

### 1. Environment Variables
Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### 2. CUDA Device Configuration
The script defaults to using GPU 1. To change the CUDA device, modify line 9 in `aviation_grpo.py`:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change to your desired GPU ID
```

### 3. Dataset
To obtain the dataset, contact the authors of this article: https://www.sciencedirect.com/science/article/pii/S0957417425000442

Request permission to use their dataset. Upon approval, they will provide you with an .xlsx file. Place your `GAHFACS.xlsx` dataset file in the project directory, or specify a custom path using the `--gahfacs-path` argument. The synthetic dataset generation algorithm samples 10 few-shot examples from the GAHFACS dataset to generate the synthetic dataset.

## Usage

### Training a Model
```bash
# Basic training with auto-generated folder name
python aviation_grpo.py --train

# Training with custom LoRA folder name
python aviation_grpo.py --train --save-lora-folder my_model

# Training with custom dataset and synthetic data paths
python aviation_grpo.py --train --gahfacs-path /path/to/GAHFACS.xlsx --save-synthetic-data synth_my_model.xlsx
```

### Evaluating a Model
```bash
# Evaluate a trained model
python aviation_grpo.py --evaluate --load-lora-folder my_model

# Evaluate with matching synthetic data
python aviation_grpo.py --evaluate --load-lora-folder my_model --load-synthetic-data synth_my_model.xlsx
```

### Training and Evaluating Together
```bash
# Train and evaluate in one command
python aviation_grpo.py --train --evaluate --save-lora-folder my_model
```

## Command Line Arguments

### Required Actions
- `--train`: Train the model using GRPO
- `--evaluate`: Evaluate the model and generate comparison graphs

### Optional Arguments
- `--save-lora-folder NAME`: LoRA folder name to save to (default: grpo_saved_YYYYMMDD_HHMMSS)
- `--load-lora-folder NAME`: LoRA folder name to load from (default: same as save folder)
- `--gahfacs-path PATH`: Path to main GAHFACS dataset (default: GAHFACS.xlsx)
- `--load-synthetic-data PATH`: Path to existing synthetic dataset to load
- `--save-synthetic-data PATH`: Custom path to save synthetic data
- `--help`: Show detailed help message

## Output Files

### Training
- LoRA model files in the specified folder
- Reward logs in multiple formats (JSONL, Parquet, Excel)
- Synthetic data files (if generated)

### Evaluation
- Timestamped graphs directory containing:
  - Performance comparison visualizations
  - Detailed evaluation results (Excel)
  - Evaluation summary (Markdown)
  - Power consumption data

## Model Comparisons

The evaluation includes comparisons with:
- Baseline model (without GRPO)
- GRPO-trained model
- External models (GPT-5-mini, Gemini-2.5-flash)
- OSS models via Ollama (gpt-oss, gemma3, deepseek-r1, etc.)

## Cost Management

The script includes automatic cost estimation and confirmation prompts for API usage for GPT-5 and Gemini. Costs are tracked and will prompt for confirmation if estimated usage exceeds $20.


## Notes

- The script requires at least 24GB GPU memory for training and evaluation
- API costs also applies for synthetic data generation and external model comparisons
- Power consumption monitoring runs during training and evaluation
- Results are automatically timestamped and organized in the graphs directory 
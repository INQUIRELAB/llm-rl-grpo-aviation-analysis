### Train and evaluate model with matching arguments

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MODEL_NAME="aviation_model_${TIMESTAMP}"

SYNTHETIC_FILE="synth_GAHFACS_${TIMESTAMP}.xlsx"

# Run the python code without any argument to see the guide of how to use the arguments

# Train the model and save synthetic data with controlled filename; also capture output
python aviation_grpo.py --train --save-lora-folder "$MODEL_NAME" --save-synthetic-data "$SYNTHETIC_FILE" | tee "train_${TIMESTAMP}.log"

# Evaluate the trained model using the same synthetic data from training; also capture output
python aviation_grpo.py --evaluate --load-lora-folder "$MODEL_NAME" --load-synthetic-data "$SYNTHETIC_FILE" | tee "eval_${TIMESTAMP}.log"

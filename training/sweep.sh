#!/usr/bin/bash

# Create a new sweep and extract its ID
SWEEP_ID=$(wandb sweep --project serimats training/l9h6_sweep_config.yaml 2>&1 | grep 'Run sweep agent with: wandb agent' | awk '{print $NF}')
echo "Created sweep with ID: $SWEEP_ID"

# Check if SWEEP_ID is empty
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create sweep or extract SWEEP_ID."
    exit 1
fi

# Launch wandb agents, each on a different GPU
for i in 0 2 3; do
    CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID --count 100 &
done

# Wait for all background processes to finish
wait
echo "All sweep agents have completed."

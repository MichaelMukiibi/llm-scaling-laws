# Flax NNX Transformer for Double-Digit Addition

An autoregressive Transformer Language Model built from scratch using JAX and the Flax NNX API, optimized to learn the algorithmic rules of double-digit addition (e.g., `45+22=67`).

## Project Features

* **Causal Transformer Architecture**: Built with multi-head attention, Pre-LayerNorm configuration, and learned positional embeddings.
* **Algorithmic Task Dataset**: Dynamically generates all double-digit addition combinations with zero-masked padding to ensure the model focuses purely on mathematical logic.
* **Optimized Execution Pipeline**: Compiled via `@nnx.jit` with strict non-blocking GPU loss accumulation to eliminate CPU-GPU synchronization overhead.
* **Experiment Management**: Integrated tracking for validation metrics and secure weight extraction uploaded directly to Weights & Biases.

## Environment Setup

Ensure the required dependencies are installed in your execution environment:

```bash
pip install jax jaxlib flax optax wandb numpy
```

## How to Run

Execute the training script within your custom environment infrastructure by passing the required configuration parameters:

### W&B API Key

```.env
WANDB_API_KEY=wandb_api_key_here
```

```bash
# Add .env file
source .env

# Run the accelerated training session on a T4 GPU instance
colab run --gpu T4 -s addition-run train.py --wandb_key $WANDB_API_KEY --epochs 20 --batch_size 64
```
## Tracked Metrics & Outputs
* **Loss Functions**: Monitors cross-entropy loss on a 90/10 train-to-validation split.

* **W&B Dashboards**: Tracks gradient progress live using the `flax-transformer-addition` project workspace.

* **Model Serialization**: Automatically extracts parameters down to a raw Python dictionary using `nnx.to_pure_dict()`, saves them as a secure `model_state.msgpack` binary file, and registers the file as a W&B Model Artifact upon completion.
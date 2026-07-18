import argparse
import jax
import jax.numpy as jnp
from flax import nnx
from flax import serialization
import wandb
import numpy as np

# Reuse the exact architecture definition from train.py
class CausalAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim).swapaxes(1, 2)
        scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(self.head_dim)
        mask = jnp.tril(jnp.ones((s, s)))
        adder = (1.0 - mask) * -1e9
        attn_weights = jax.nn.softmax(scores + adder, axis=-1)
        context = jnp.matmul(attn_weights, v).swapaxes(1, 2)
        return self.out_proj(context.reshape(b, s, d))

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, ffn_dim: int, num_heads: int, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.attn = CausalAttention(d_model, num_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.fc1 = nnx.Linear(in_features=d_model, out_features=ffn_dim, rngs=rngs)
        self.fc2 = nnx.Linear(in_features=ffn_dim, out_features=d_model, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(jax.nn.relu(self.fc1(self.ln2(x))))
        return x

class TransformerLM(nnx.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int, ffn_dim: int, num_heads: int, num_layers: int, rngs: nnx.Rngs):
        self.token_embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_embed = nnx.Embed(max_len, d_model, rngs=rngs)
        self.blocks = [
            TransformerBlock(d_model, ffn_dim, num_heads, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.ln_f = nnx.LayerNorm(d_model, rngs=rngs)
        self.lm_head = nnx.Linear(d_model, vocab_size, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, s = x.shape
        h = self.token_embed(x)
        positions = jnp.arange(s)
        h = h + self.pos_embed(positions)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.ln_f(h))

# Setup dictionary mappings
vocab = "0123456789+=P"
char2id = {c: i for i, c in enumerate(vocab)}
id2char = {i: c for i, c in enumerate(vocab)}

@nnx.jit
def sample_next_token(model: TransformerLM, x: jnp.ndarray, max_len: int) -> jnp.ndarray:
    logits = model(x)
    # Isolate the final token logits and greedily decode
    next_token_logits = logits[:, -1, :]
    return jnp.argmax(next_token_logits, axis=-1, keepdims=True)

def main():
    parser = argparse.ArgumentParser(description="Inference Script for Addition Transformer")
    parser.add_argument("--wandb_key", type=str, required=True, help="W&B API Key")
    parser.add_argument("--artifact_path", type=str, required=True, help="W&B artifact path (entity/project/name:version)")
    parser.add_argument("--prompt", type=str, default="45+22=", help="Addition prompt sequence")
    args = parser.parse_args()

    # Authenticate and fetch the weights file from W&B
    wandb.login(key=args.wandb_key)
    run = wandb.init(project="flax-transformer-addition-eval", job_type="inference")
    artifact = run.use_artifact(args.artifact_path, type="model")
    artifact_dir = artifact.download()
    
    # Reinitialize an identical architecture skeleton
    rngs = nnx.Rngs(2)
    max_len = 16
    model = TransformerLM(vocab_size=len(vocab), max_len=max_len, d_model=128, ffn_dim=512, num_heads=4, num_layers=3, rngs=rngs)

    # Read binary bytes and push raw dictionary values directly back into model state
    with open(f"{artifact_dir}/model_state.msgpack", "rb") as f:
        state_bytes = f.read()
    
    target_dict = nnx.to_pure_dict(nnx.state(model))
    pure_dict = serialization.from_bytes(target_dict, state_bytes)

    # Convert the pure dictionary into an NNX State container and update the model parameters
    nnx.update(model, nnx.State(pure_dict))
    print("Model parameters successfully restored.")

    # Encode prompt string into JAX tokens
    input_ids = [char2id[c] for c in args.prompt]
    x = jnp.array([input_ids], dtype=jnp.int32)

    # Run autoregressive loop generation
    print(f"\nPrompt: {args.prompt}")
    while x.shape[1] < max_len:
        # Enforce the max context length limitation here on the host (CPU)
        x_input = x[:, -max_len:]

        # Execute the JIT forward pass with a predictable static shape size
        next_token = sample_next_token(model, x, max_len)
        x = jnp.concatenate([x, next_token], axis=1)
        
        # Stop early if padding tokens or terminal structures are hit
        char = id2char[int(next_token[0, 0])]
        if char == 'P':
            break

    # Reconstruct whole decoded sequence
    full_sequence = "".join([id2char[int(idx)] for idx in np.array(x[0])])
    print(f"Model Output: {full_sequence.replace('P', '')}")
    wandb.finish()

if __name__ == "__main__":
    main()
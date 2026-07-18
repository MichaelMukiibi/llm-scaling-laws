import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
import numpy as np

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

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

# ==========================================
# 2. DATASET GENERATION
# ==========================================

def generate_addition_dataset():
    vocab = "0123456789+=P"
    char2id = {c: i for i, c in enumerate(vocab)}
    
    x_data, y_data = [], []
    for a in range(10, 100):
        for b in range(10, 100):
            eq = f"{a}+{b}={a+b}"
            eq = eq.ljust(17, 'P') 
            encoded = [char2id[c] for c in eq]
            x_data.append(encoded[:-1])
            y_data.append(encoded[1:])
            
    x_arr = np.array(x_data)
    y_arr = np.array(y_data)
    
    np.random.seed(42)
    indices = np.random.permutation(len(x_arr))
    
    return x_arr[indices], y_arr[indices], len(vocab)

# ==========================================
# 3. TRAINING FUNCTIONS
# ==========================================

def loss_fn(model: TransformerLM, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = model(x)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(token_losses)

@nnx.jit
def train_step(model: TransformerLM, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM on Addition")
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API Key")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    x_data, y_data, vocab_size = generate_addition_dataset()
    max_len = x_data.shape[1]

    wandb.login(key=args.wandb_key)
    wandb.init(
        project="flax-transformer-addition",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": 128,
            "vocab_size": vocab_size,
            "max_len": max_len,
            "learning_rate": 1e-3,
            "ffn_dim": 512,
            "num_heads": 4,
            "num_layers": 3
        }
    )

    rngs = nnx.Rngs(42)
    model = TransformerLM(
        vocab_size=wandb.config.vocab_size, 
        max_len=wandb.config.max_len, 
        d_model=wandb.config.d_model, 
        ffn_dim=wandb.config.ffn_dim, 
        num_heads=wandb.config.num_heads, 
        num_layers=wandb.config.num_layers, 
        rngs=rngs
    )
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=wandb.config.learning_rate), wrt=nnx.Param)

    steps_per_epoch = len(x_data) // args.batch_size

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            start_idx = i * args.batch_size
            end_idx = start_idx + args.batch_size
            
            x_batch = jnp.array(x_data[start_idx:end_idx])
            y_batch = jnp.array(y_data[start_idx:end_idx])
            
            # ISSUE: Calling .item() here forces a slow CPU-GPU sync every single step
            loss = train_step(model, optimizer, x_batch, y_batch)
            epoch_loss += loss.item()
            
            wandb.log({"step_loss": loss.item()})
            
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

    from flax import serialization

    # Extract the tracked parameter state from the Flax NNX model
    model_state = nnx.state(model)

    # Unpack the custom State object into a raw Python dict of JAX arrays
    pure_state_dict = nnx.to_pure_dict(model_state)

    # Serialize the state PyTree into a secure msgpack byte stream
    state_bytes = serialization.to_bytes(pure_state_dict)

    # Save the serialized bytes to a local file
    weights_path = "model_state.msgpack"
    with open(weights_path, "wb") as f:
        f.write(state_bytes)

    # Initialize the W&B artifact container
    artifact = wandb.Artifact(
        name="transformer-addition-weights",
        type="model",
        description="Native Flax NNX model weights saved via msgpack serialization."
    )

    # Attach the file and push it to the active project run
    artifact.add_file(weights_path)
    wandb.log_artifact(artifact)

    print("Model saved.")

    wandb.finish()

    print("Training complete.")



if __name__ == "__main__":
    main()
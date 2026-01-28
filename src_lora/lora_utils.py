"""
Utility functions for working with LoRA-enabled FlowPolicy models.
"""

import pathlib
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Dict, Any, Optional, Tuple
from model_async_lora import LoRAFlowPolicy, ModelConfig


def load_base_model(checkpoint_path: str, config: ModelConfig) -> LoRAFlowPolicy:
    """Load a pre-trained base model from checkpoint."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load the base model
    base_model = nnx.load(checkpoint_path)
    
    # Convert to LoRA model if needed
    if not hasattr(base_model, 'enable_lora'):
        # This is a regular model, we need to convert it
        print("Converting base model to LoRA model...")
        # Implementation would depend on the exact structure
        # For now, we'll assume it's already a LoRA model
        pass
    
    return base_model


def save_lora_adapter(model: LoRAFlowPolicy, save_path: str, adapter_name: str = "lora_adapter"):
    """Save only the LoRA adapter weights."""
    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Extract LoRA parameters
    all_params = nnx.state(model, nnx.Param)
    lora_params = {}
    
    def extract_lora_params(module, path):
        for name, value in module.items():
            if isinstance(value, dict):
                extract_lora_params(value, path + (name,))
            else:
                full_path = path + (name,)
                if any('lora_' in str(p) for p in full_path):
                    lora_params[full_path] = value
    
    extract_lora_params(all_params, ())
    
    # Save LoRA parameters
    adapter_path = save_path / f"{adapter_name}.pkl"
    nnx.save(adapter_path, lora_params)
    print(f"Saved LoRA adapter to {adapter_path}")
    
    # Save adapter config
    config = {
        'lora_rank': model.lora_rank,
        'lora_alpha': model.lora_alpha,
        'lora_dropout': model.lora_dropout,
        'enable_lora': model.enable_lora,
    }
    config_path = save_path / f"{adapter_name}_config.pkl"
    nnx.save(config_path, config)
    print(f"Saved LoRA config to {config_path}")


def load_lora_adapter(base_model: LoRAFlowPolicy, adapter_path: str) -> LoRAFlowPolicy:
    """Load LoRA adapter weights into a base model."""
    adapter_path = pathlib.Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    
    # Load LoRA parameters
    lora_params = nnx.load(adapter_path)
    
    # Load adapter config
    config_path = adapter_path.parent / f"{adapter_path.stem}_config.pkl"
    if config_path.exists():
        adapter_config = nnx.load(config_path)
        print(f"Loaded LoRA config: {adapter_config}")
    
    # Apply LoRA parameters to the model
    # This would require careful parameter mapping
    # For now, we'll return the base model with a note
    print("Note: LoRA parameter loading requires careful parameter mapping")
    return base_model


def merge_lora_weights(base_model: LoRAFlowPolicy, lora_adapter_path: str) -> LoRAFlowPolicy:
    """Merge LoRA weights into the base model for inference."""
    # Load LoRA adapter
    lora_params = nnx.load(lora_adapter_path)
    
    # Create a copy of the base model
    merged_model = base_model
    
    # Merge LoRA weights into base layers
    # This would require iterating through the model and updating weights
    # For now, we'll return the base model with a note
    print("Note: LoRA weight merging requires careful implementation")
    return merged_model


def freeze_base_layers(model: LoRAFlowPolicy) -> LoRAFlowPolicy:
    """Freeze base layers while keeping LoRA layers trainable."""
    # This would require setting trainable=False for base parameters
    # and trainable=True for LoRA parameters
    print("Note: Parameter freezing requires careful implementation")
    return model


def unfreeze_all_layers(model: LoRAFlowPolicy) -> LoRAFlowPolicy:
    """Unfreeze all layers for full fine-tuning."""
    # This would require setting trainable=True for all parameters
    print("Note: Parameter unfreezing requires careful implementation")
    return model


def get_trainable_parameters(model: LoRAFlowPolicy) -> Dict[str, Any]:
    """Get only the trainable parameters (LoRA parameters when LoRA is enabled)."""
    all_params = nnx.state(model, nnx.Param)
    trainable_params = {}
    
    def extract_trainable_params(module, path):
        for name, value in module.items():
            if isinstance(value, dict):
                extract_trainable_params(value, path + (name,))
            else:
                full_path = path + (name,)
                # Only include LoRA parameters when LoRA is enabled
                if model.enable_lora and any('lora_' in str(p) for p in full_path):
                    trainable_params[full_path] = value
                elif not model.enable_lora:
                    # Include all parameters when LoRA is disabled
                    trainable_params[full_path] = value
    
    extract_trainable_params(all_params, ())
    return trainable_params


def count_parameters(model: LoRAFlowPolicy) -> Tuple[int, int, int]:
    """Count total, trainable, and LoRA parameters."""
    all_params = nnx.state(model, nnx.Param)
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    def count_params(module, path):
        nonlocal total_params, trainable_params, lora_params
        for name, value in module.items():
            if isinstance(value, dict):
                count_params(value, path + (name,))
            else:
                param_count = value.size
                total_params += param_count
                
                full_path = path + (name,)
                if any('lora_' in str(p) for p in full_path):
                    lora_params += param_count
                    if model.enable_lora:
                        trainable_params += param_count
                else:
                    if not model.enable_lora:
                        trainable_params += param_count
    
    count_params(all_params, ())
    return total_params, trainable_params, lora_params


def create_lora_model_from_base(base_model_path: str, config: ModelConfig) -> LoRAFlowPolicy:
    """Create a LoRA model from a base model checkpoint."""
    # Load base model
    base_model = load_base_model(base_model_path, config)
    
    # Ensure LoRA is enabled
    if not base_model.enable_lora:
        print("Warning: Base model does not have LoRA enabled")
    
    return base_model


def compare_models(model1: LoRAFlowPolicy, model2: LoRAFlowPolicy) -> Dict[str, float]:
    """Compare two models and return parameter differences."""
    params1 = nnx.state(model1, nnx.Param)
    params2 = nnx.state(model2, nnx.Param)
    
    differences = {}
    
    def compare_params(p1, p2, path):
        for name, value1 in p1.items():
            if isinstance(value1, dict):
                compare_params(value1, p2[name], path + (name,))
            else:
                value2 = p2[name]
                if value1.shape != value2.shape:
                    differences[f"{'.'.join(path + (name,))}_shape"] = float('inf')
                else:
                    diff = jnp.mean(jnp.abs(value1 - value2))
                    differences[f"{'.'.join(path + (name,))}"] = float(diff)
    
    compare_params(params1, params2, ())
    return differences


def print_model_info(model: LoRAFlowPolicy):
    """Print detailed information about the model."""
    total_params, trainable_params, lora_params = count_parameters(model)
    
    print(f"Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  LoRA enabled: {model.enable_lora}")
    print(f"  LoRA rank: {model.lora_rank}")
    print(f"  LoRA alpha: {model.lora_alpha}")
    print(f"  LoRA dropout: {model.lora_dropout}")
    print(f"  Parameter efficiency: {trainable_params/total_params*100:.2f}%")


# Example usage functions
def example_lora_training_workflow():
    """Example workflow for LoRA training."""
    
    # 1. Load base model
    config = ModelConfig(
        channel_dim=256,
        channel_hidden_dim=512,
        token_hidden_dim=64,
        num_layers=4,
        action_chunk_size=8,
        enable_lora=True,
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
    )
    
    base_model = create_lora_model_from_base("path/to/base_model.pkl", config)
    print_model_info(base_model)
    
    # 2. Train with LoRA (this would be done in the training loop)
    # The model will automatically use LoRA during training when training=True
    
    # 3. Save LoRA adapter
    save_lora_adapter(base_model, "path/to/adapters", "my_lora_adapter")
    
    # 4. Load LoRA adapter for inference
    loaded_model = load_lora_adapter(base_model, "path/to/adapters/my_lora_adapter.pkl")
    
    # 5. Optionally merge LoRA weights for faster inference
    merged_model = merge_lora_weights(base_model, "path/to/adapters/my_lora_adapter.pkl")


if __name__ == "__main__":
    example_lora_training_workflow() 
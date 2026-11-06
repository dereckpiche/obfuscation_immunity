import pytest
import torch


@pytest.fixture
def actor_model():
    """Builds a small random Llama with a LoRA adapter, as verl's actor
    holds it.
    """

    def build(seed=0):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(seed)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        model = get_peft_model(
            LlamaForCausalLM(config),
            LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"),
        )
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter)
        return model

    return build

import os

import torch
from torch import nn
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, RuntimeError):
    from tensorboardX import SummaryWriter


BASE_MODEL_PATH = "/Users/fajri/Projects/mytestmodel"
DATASET_PATH = (
    "/Users/fajri/Projects/testmodel/step4/dataset/dataset.jsonl"
)
OUTPUT_PATH = "/Users/fajri/Projects/testmodel/step7/output"
TENSORBOARD_LOG_PATH = (
    "/Users/fajri/Projects/testmodel/step7/tensorboard_logs"
)
NEURAL_NETWORK_LOG_PATH = (
    "/Users/fajri/Projects/testmodel/"
    "step7/tensorboard_logs/base_model"
)


class DecoderLayerGraphView(nn.Module):
    """Ringkasan visual satu decoder layer untuk TensorBoard Graphs."""

    def __init__(self, layer):
        super().__init__()
        self.q_proj = layer.self_attn.q_proj
        self.k_proj = layer.self_attn.k_proj
        self.v_proj = layer.self_attn.v_proj
        self.o_proj = layer.self_attn.o_proj
        self.gate_proj = layer.mlp.gate_proj
        self.up_proj = layer.mlp.up_proj
        self.down_proj = layer.mlp.down_proj
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        # Jalur ringkas ini dibuat untuk visualisasi struktur. Ini bukan
        # pengganti forward attention asli pada saat inference/training.
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        key_value = torch.cat((key, value), dim=-1)
        attention = self.o_proj(query + key_value)

        gate = self.activation(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        mlp = self.down_proj(gate * up)

        return hidden_states + attention + mlp


class ModelGraphView(nn.Module):
    """Tampilan ringkas arsitektur model untuk TensorBoard Graphs."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(
            DecoderLayerGraphView(layer)
            for layer in model.model.layers
        )
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def report_neural_network(model, tokenizer):
    print("Merekam struktur neural network ke TensorBoard...")
    writer = SummaryWriter(
        log_dir=NEURAL_NETWORK_LOG_PATH,
    )

    try:
        writer.add_text(
            "Neural Network/Base Model",
            f"```\n{model}\n```",
            global_step=0,
        )

        graph_input = tokenizer(
            "Saya terkena bisa",
            return_tensors="pt",
        )["input_ids"].to(next(model.parameters()).device)

        graph_view = ModelGraphView(model).eval()
        writer.add_graph(
            graph_view,
            graph_input,
            use_strict_trace=False,
        )
    finally:
        writer.close()

    print(
        "Struktur neural network disimpan di: "
        f"{NEURAL_NETWORK_LOG_PATH}"
    )


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.float32,
    attn_implementation="eager",
    local_files_only=True,
)

base_model.config.use_cache = False

# Struktur neural network base model direkam sebelum LoRA dipasang.
report_neural_network(base_model, tokenizer)

if os.environ.get("GRAPH_ONLY") == "1":
    print("GRAPH_ONLY=1, training LoRA tidak dijalankan.")
    raise SystemExit(0)

base_model.train()
base_model.enable_input_require_grads()

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

training_config = SFTConfig(
    output_dir=OUTPUT_PATH,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=256,
    logging_steps=1,
    save_strategy="epoch",
    gradient_checkpointing=True,
    report_to="tensorboard",
    logging_dir=TENSORBOARD_LOG_PATH,
)

trainer = SFTTrainer(
    model=base_model,
    args=training_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()
trainer.train()

trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Adapter disimpan di: {OUTPUT_PATH}")

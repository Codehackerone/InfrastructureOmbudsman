# BERT and RoBERTa Hyperparameters
## Number of training epochs
num_train_epochs = 5
## Batch size per GPU for training
per_device_train_batch_size = 32
## Batch size per GPU for evaluation
per_device_eval_batch_size = 32
> Everything else is the default Huggingface Training Arguments

# LLAMA 2 and Mistral Hyperparameters
## LoRA attention dimension
lora_r = 64
## Alpha parameter for LoRA scaling
lora_alpha = 16
## Dropout probability for LoRA layers
lora_dropout = 0.4
## Activate 4-bit precision base model loading
use_4bit = True
## Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
## Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
## Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
## Number of training epochs
num_train_epochs = 5
## Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
## Batch size per GPU for training
per_device_train_batch_size = 8
## Batch size per GPU for evaluation
per_device_eval_batch_size = 8
## Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
## Enable gradient checkpointing
gradient_checkpointing = True
## Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
## Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
## Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
## Optimizer to use
optim = "paged_adamw_32bit"
## Learning rate schedule
lr_scheduler_type = "cosine"
## Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
## Group sequences into batches with same length. Saves memory and speeds up training considerably
group_by_length = True

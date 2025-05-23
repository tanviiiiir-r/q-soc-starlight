from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from datasets import load_dataset

model_id = "meta-llama/Llama-2-7b-hf"
dataset = load_dataset("json", data_files="mitre_prompts.jsonl")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def tokenize_fn(examples):
    return tokenizer(examples["prompt"] + " " + examples["completion"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    output_dir="./llama2_mitre",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=100,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

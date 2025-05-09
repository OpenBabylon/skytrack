"""Minimal HuggingFace Trainer example with SkyTrack callback."""
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import skytrack as st

cfg = {"project": "skytrack-hf", "run_name": "sst2-quick"}
st.init(cfg)

dataset = load_dataset("glue", "sst2", split="train[:2%]").train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tok(ex): return tokenizer(ex["sentence"], truncation=True, padding="max_length")
dataset = dataset.map(tok, batched=True)
dataset = dataset.remove_columns(["sentence", "idx"]).rename_column("label", "labels")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

args = TrainingArguments(
    output_dir="./out",
    evaluation_strategy="epoch",
    logging_steps=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    report_to=["none"],   # HF's wandb reporter disabled; we use callback
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    callbacks=[st.SkyTrackCallback()],
)

trainer.train()

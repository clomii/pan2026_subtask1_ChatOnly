import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

def load_jsonl(filepath):
    texts, labels, ids = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                texts.append(obj['text'])
                labels.append(obj['label'])
                ids.append(obj['id'])
    return texts, labels, ids

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # DeBERTa outputs logits, we want the argmax for accuracy/F1
    preds = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

def main():
    MODEL_NAME = "microsoft/deberta-v3-small"
    
    print("Loading datasets...")
    train_texts, train_labels, train_ids = load_jsonl('data/train.jsonl')
    val_texts, val_labels, val_ids = load_jsonl('data/val.jsonl')
    
    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    
    print(f"Loading Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
    print("Tokenizing datasets (this may take a minute)...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    print("Loading Model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Define training arguments optimized for a 12GB VRAM GPU (RTX 4070 Ti)
    training_args = TrainingArguments(
        output_dir="./deberta_results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2, # Effectively gives batch size 16 to save VRAM
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True, # Critical for 4070 Ti to save VRAM and speed up training!
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting Training (This will take a while!)...")
    trainer.train()
    
    print("Generating validation predictions...")
    predictions = trainer.predict(tokenized_val)
    
    # Convert logits to probabilities using Softmax
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    
    print("Saving to deberta_model.jsonl...")
    with open('deberta_model.jsonl', 'w', encoding='utf-8') as f:
        for i, doc_id in enumerate(val_ids):
            f.write(json.dumps({"id": doc_id, "label": float(probs[i])}) + "\n")
            
    print("Done! You now have a Deep Learning Semantic model output.")

if __name__ == '__main__':
    main()

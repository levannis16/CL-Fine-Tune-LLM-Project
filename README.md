# CL-Fine-Tune-LLM-Project
This Project is under course Computational Linguistics KMK3053 UNIMAS.

from datasets import load_dataset

print("Loading SQuAD v2 dataset...")
squad = load_dataset("squad_v2")

print(squad)
print(squad["train"][0])

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
print(f"Loading model and tokenizer: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

pip install transformers

from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
print(f"Loading model and tokenizer: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

def preprocess_data(examples):
    # Tokenize the input data
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        max_length=384,
        stride=128,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
  # Initialize start and end positions
  start_positions = []
  end_positions = []
    
  for i in range(len(examples["answers"]["answer_start"])):
        if len(examples["answers"]["answer_start"][i]) > 0:  # Check if an answer exists
            start_pos = examples["answers"]["answer_start"][i][0]
            end_pos = start_pos + len(examples["answers"]["text"][i][0])
        else:  # For unanswerable questions, use default values
            start_pos = 0
            end_pos = 0
        
  start_positions.append(start_pos)
  end_positions.append(end_pos)
    
  tokenized["start_positions"] = start_positions
  tokenized["end_positions"] = end_positions
  return tokenized

print("Preprocessing the dataset...")
tokenized_data = squad.map(preprocess_data, batched=True)

def to_tf_dataset(data):
    return tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"]
        },
        {
            "start_positions": data["start_positions"],
            "end_positions": data["end_positions"]
        }
    )).batch(16)

train_dataset = to_tf_dataset(tokenized_data["train"])
val_dataset = to_tf_dataset(tokenized_data["validation"])

print("Compiling the model...")
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss={"start_positions": loss, "end_positions": loss})

print("Training the model...")
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

print("Evaluating the model...")
results = model.evaluate(val_dataset)
print(f"Evaluation results: {results}")

print("Saving the fine-tuned model...")
model.save_pretrained("./fine_tuned_qa_model")
tokenizer.save_pretrained("./fine_tuned_qa_model")

print("Model fine-tuning completed and saved.")


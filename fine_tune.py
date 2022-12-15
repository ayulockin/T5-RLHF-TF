import wandb
from wandb.keras import WandbMetricsLogger

import tensorflow as tf

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM
from transformers import create_optimizer, AdamWeightDecay

use_wandb = True

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


PREFIX = "summarize: "

# Get the dataset and split it
billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)

# Get the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Get the model
model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.summary()

# Build dataloaders
# Preprocessing function to append the PREFIX to prompt the model for summarization task.
def preprocess_function(examples):
    inputs = [PREFIX + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=48, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")

tf_train_set = model.prepare_tf_dataset(
    tokenized_billsum["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_billsum["test"],
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator,
)

# Compile the model
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer, jit_compile=True)

# Initialize a W&B run
callbacks = []
if use_wandb:
    run = wandb.init(
        project="rlhf-tf",
    )

    callbacks += [WandbMetricsLogger()]

# Train the model
model.fit(
    x=tf_train_set,
    validation_data=tf_test_set,
    epochs=3,
    callbacks=callbacks
)

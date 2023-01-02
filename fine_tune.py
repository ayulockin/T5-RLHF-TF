import string
import random
import wandb
from wandb.keras import WandbMetricsLogger

import tensorflow as tf

from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM

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


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

exp_id = id_generator(size=8)
print('Experiment Id: ', exp_id)


USE_WANDB = False
BATCH_SIZE = 8
PREFIX = "summarize: "
DATA_PATH = "../dataset_splits"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 1


# Get the dataset
train_dataset = load_dataset("json", data_files=f"{DATA_PATH}/train_0_1.jsonl")["train"]
valid_dataset = load_dataset("json", data_files=f"{DATA_PATH}/valid_0_1.jsonl")["train"]

# Remove unwanted columns
train_dataset = train_dataset.remove_columns(['id', 'subreddit', 'title'])
valid_dataset = valid_dataset.remove_columns(['id', 'subreddit', 'title'])

# Get the model
tf.keras.backend.clear_session()
model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")
model.summary()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")


# Preprocess the text
def preprocess_function(examples):
    inputs = [PREFIX + doc for doc in examples["post"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_target=examples["summary"], max_length=48, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Prepare dataloader
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
tf_train_set = model.prepare_tf_dataset(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

tf_valid_set = model.prepare_tf_dataset(
    valid_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)

# Optimizer and compile
optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
model.compile(optimizer=optimizer)

# Callbacks
callbacks = []

# Initialize a W&B run
if USE_WANDB:
    run = wandb.init(
        project="rlhf-tf",
        job_type="fine_tune"
    )

    callbacks += [WandbMetricsLogger()]

# Train the model
model.fit(
    x=tf_train_set,
    validation_data=tf_valid_set,
    epochs=EPOCHS,
    callbacks=[]
)

# Save the model for inference
model.save_pretrained(f"models/model_{exp_id}", save_model=True)

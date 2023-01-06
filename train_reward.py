import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import string
import random
from tqdm import tqdm

import wandb
from wandb.keras import WandbMetricsLogger

import tensorflow as tf

from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TFAutoModelForSeq2SeqLM

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_tf_t5 import TFT5Model
from transformers.tf_utils import shape_list


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

exp_id = id_generator(size=8) # TODO: lowercase
print('Experiment Id: ', exp_id)


USE_WANDB = False
BATCH_SIZE = 32
PREFIX = "binary classification: "
DATA_PATH = "../comparison_splits"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
AUTOTUNE = tf.data.AUTOTUNE


# Get the dataset
train_dataset = load_dataset("json", data_files=f"{DATA_PATH}/train.jsonl")["train"]
valid_dataset = load_dataset("json", data_files=f"{DATA_PATH}/valid.jsonl")["train"]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Preprocess raw text
def preprocess_function(examples):
    inputs = [post + "\n" + summary for post, summary in zip(examples["post"], examples["summary"])]
    inputs = [PREFIX + doc for doc in examples["post"]]
    model_inputs = tokenizer(inputs, max_length=550, truncation=True)

    # Setup the tokenizer for targets
    labels = examples["label"]
    labels = [str(label) for label in labels]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_target=labels, max_length=3, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["choice"] = examples["label"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
valid_dataset = valid_dataset.map(preprocess_function, batched=True)

# Remove unwanted columns
train_dataset = train_dataset.remove_columns(['id', 'post', 'summary', 'label'])
valid_dataset = valid_dataset.remove_columns(['id', 'post', 'summary', 'label'])

# Prepare dataloaders
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="tf")
trainloader = train_dataset.to_tf_dataset(
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator,
            prefetch=False
        )

validloader = valid_dataset.to_tf_dataset(
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator,
            prefetch=False
        )


def shift_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert decoder_start_token_id is not None, (
        "self.model.config.decoder_start_token_id has to be defined. In TF T5 it is usually set to the"
        " pad_token_id. See T5 docs for more information"
    )

    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    start_tokens = tf.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = tf.where(
        shifted_input_ids == -100,
        tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
        shifted_input_ids,
    )

    # "Verify that `labels` has only positive values and -100"
    assert_gte0 = tf.debugging.assert_greater_equal(
        shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype)
    )

    # Make sure the assertion op is called by wrapping the result in an identity no-op
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


# @tf.function(jit_compile=True)
def parse_data(inputs):
    # This will be used as decoder_input_ids
    shift_right_labels = shift_right(inputs["labels"])
    inputs["decoder_input_ids"] = shift_right_labels
    inputs.pop("labels")

    # This label is for calculating the accuracy of the reward model
    labels = inputs["choice"]
    inputs.pop("choice")

    return inputs, labels


trainloader = trainloader.map(parse_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
validloader = validloader.map(parse_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


class T5RewardModel(tf.keras.Model):
    def __init__(self):
        super(T5RewardModel, self).__init__()
        config = T5Config()
        self.t5_without_lm_head = TFT5Model(config).from_pretrained("t5-small")
        self.reward_head = tf.keras.layers.Dense(1, use_bias=False, activation="sigmoid")

    def call(self, inputs, training=False):
        sequence_output = self.t5_without_lm_head(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=inputs["decoder_input_ids"],
            training=training).last_hidden_state

        reward_output = self.reward_head(sequence_output[:,0,:]) ## extract the 1st token's embeddings

        return reward_output

# Optimizer
optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Initialize model and compile it
tf.keras.backend.clear_session()
reward_model = T5RewardModel()
reward_model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = []

# Initialize a W&B run
if USE_WANDB:
    run = wandb.init(
        project="rlhf-tf",
        job_type="reward_training"
    )

    callbacks += [WandbMetricsLogger()]

# Train the model
reward_model.fit(
    x=trainloader,
    validation_data=validloader,
    epochs=EPOCHS,
    callbacks=callbacks
)

# # Save the model for inference
# model.save_pretrained(f"models/reward_model_{exp_id}", save_model=True)



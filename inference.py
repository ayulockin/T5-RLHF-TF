import wandb
from wandb.keras import WandbMetricsLogger

import tensorflow as tf
from tqdm import tqdm

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


USE_WANDB = False
BATCH_SIZE = 8
PREFIX = "summarize: "
DATA_PATH = "../dataset_splits"
MODEL_PATH = "models/model_X3HQ1BXZ"


test_dataset = load_dataset("json", data_files=f"{DATA_PATH}/test_0_1.jsonl")["train"]
# test_dataset = test_dataset.remove_columns(['id', 'subreddit', 'title'])

# Load model
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.summary()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")


@tf.function(jit_compile=True)
def generate(inputs):
    return model.generate(**inputs, max_new_tokens=48, do_sample=False)


pred_summaries = []
for i, sample in tqdm(enumerate(test_dataset)):
    post = sample["post"]
    summary = sample["summary"]

    # preprocess the post
    post = PREFIX + post
    inputs = tokenizer(post, return_tensors="tf", pad_to_multiple_of=8)

    pred_summary = generate(inputs)
    pred_summary = tokenizer.decode(pred_summary[0], skip_special_tokens=True)
    pred_summaries.append(pred_summary)
    
print(len(pred_summaries))
print(pred_summaries[:10])

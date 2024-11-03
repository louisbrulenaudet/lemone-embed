from datasets import load_dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import CachedGISTEmbedLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base", token="hf_", 
    trust_remote_code=True
)

guide = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base", token="hf_", 
    trust_remote_code=True
)

# 3. Load a dataset to finetune on
dataset = load_from_disk("lemone-training-data-max")

# 4. Define a loss function
loss = CachedGISTEmbedLoss(
    model, 
    guide, 
    mini_batch_size=128
)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="models/lemone-gte-triplet",
    num_train_epochs=1,
    per_device_train_batch_size=1024,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=200,
    run_name="lemone-gte-triplet",
)

# 6. (Optional) Create an evaluator & evaluate the base model
eval_dataset = (
    load_dataset(
        "louisbrulenaudet/tax-retrieval-benchmark",
        split="train",
        token="hf_",
    )
)

eval_corpus = {str(i): doc for i, doc in enumerate(eval_dataset["positive"])}
eval_queries = {str(i): query for i, query in enumerate(eval_dataset["query"])}
eval_relevant_docs = {str(i): {str(i)} for i in range(len(eval_dataset))}

evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    name="Lemone"
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss,
    evaluator=evaluator,
)

trainer.train()

# 8. Save the trained model
model.save_pretrained("models/lemone-gte-triplet/final")

# 9. (Optional) Push it to the Hugging Face Hub
model.push_to_hub(
    "louisbrulenaudet/lemone-gte-embed-max",
    private=True,
    token="hf_",
)

results = evaluator(model)

print(results)
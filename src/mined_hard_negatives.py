from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
# Load a Sentence Transformer model
guide_model = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base", 
    token="hf_", 
    trust_remote_code=True
)

dataset = load_dataset("./lemone-training-data")

dataset_mined = mine_hard_negatives(
    dataset=dataset,
    model=guide_model,
    range_min=10,
    range_max=65,
    max_score=0.85,
    margin=0.1,
    use_faiss=True,
    num_negatives=5,
    sampling_strategy="random",
)

dataset_mined.save_to_disk("./lemone-training-data-max")

dataset_mined_bis = mine_hard_negatives(
    dataset=dataset,
    model=guide_model,
    range_min=10,
    range_max=65,
    max_score=0.85,
    margin=0.1,
    use_faiss=True,
    num_negatives=5,
    sampling_strategy="random",
    as_triplets=False
)

dataset_mined_bis.save_to_disk("lemone-training-data-max-inline")

dataset_mined = mine_hard_negatives(
    dataset=dataset,
    model=guide_model,
    range_min=10,
    range_max=60,
    max_score=0.8,
    margin=0.1,
    use_faiss=True,
    num_negatives=5,
    sampling_strategy="random",
)
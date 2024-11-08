<img src="https://huggingface.co/louisbrulenaudet/lemone-embed-pro/resolve/main/assets/thumbnail.webp">

# Lemone-Embed: A Series of Fine-Tuned Embedding Models for French Taxation
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue)

<div class="not-prose bg-gradient-to-r from-gray-50-to-white text-gray-900 border" style="border-radius: 8px; padding: 0.5rem 1rem;">
    <p>This series is made up of 7 models, 3 basic models of different sizes trained on 1 epoch, 3 models trained on 2 epochs making up the Boost series and a Pro model with a non-Roberta architecture.</p>
</div>

This sentence transformers model, specifically designed for French taxation, has been fine-tuned on a dataset comprising 43 million tokens, integrating a blend of semi-synthetic and fully synthetic data generated by GPT-4 Turbo and Llama 3.1 70B, which have been further refined through evol-instruction tuning and manual curation. 

The model is tailored to meet the specific demands of information retrieval across large-scale tax-related corpora, supporting the implementation of production-ready Retrieval-Augmented Generation (RAG) applications. Its primary purpose is to enhance the efficiency and accuracy of legal processes in the taxation domain, with an emphasis on delivering consistent performance in real-world settings, while also contributing to advancements in legal natural language processing research.

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) <!-- at revision 7fc06782350c1a83f88b15dd4b38ef853d3b8503 -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
- **Developed by:** Louis Brulé Naudet
- **Funded by:** Microsoft for Startups
- **Shared by:** Louis Brulé Naudet
- **Model type:** Sentence Transformers
- **Language(s) (NLP):** FR
- **License:** Apache 2
- **Finetuned from model:** [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False}) with Transformer model: NewModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("louisbrulenaudet/lemone-gte-embed-max")
# Run inference
sentences = [
    "Exposer les modalités de dérogation au secret fiscal autorisant le juge à demander des documents fiscaux nécessaires pour résoudre un litige, en vertu de l'article L. 143 du Livre des Procédures Fiscales.",
    "Conformément aux dispositions de l'article L. 143 du Livre des Procédures Fiscales, le secret fiscal peut être levé dans le cadre d'un litige par décision du juge. Cette mesure vise à autoriser la présentation de documents fiscaux, jugés utiles par le magistrat pour trancher une affaire. La levée de ce secret est toutefois soumise à une interprétation stricte, de sorte que seuls les documents réellement susceptibles d'éclairer le juge sur l'étendue du préjudice des individus impliqués peuvent être divulgués. Les renseignements qui n'ont de pertinence que pour des questions périphériques de la procédure ou qui se rapportent uniquement à l'application d'un jugement déjà prononcé sont exclus de cette possibilité de communication.",
    "Selon les dispositions du Bulletin officiel des finances publiques-instructions administratives, spécifiquement le BOI-DJC-SECR-10-20-50, le procureur de la République détient le droit, dans le contexte de toute investigation judiciaire, qu'elle relève d'une enquête de flagrance, préliminaire ou autre, de solliciter des renseignements ou documents essentiels à l'enquête auprès de l'administration fiscale. Cette sollicitation peut être adressée directement ou via un officier de police judiciaire agissant sur une réquisition du procureur. Conformément à l'article L.141 A du Livre des procédures fiscales, le secret fiscal ne constitue pas un frein légal à la transmission des informations ou documents exigés par le procureur.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval
* Dataset: `Lemone`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.9737     |
| cosine_accuracy@3   | 0.9917     |
| cosine_accuracy@5   | 0.9936     |
| cosine_accuracy@10  | 0.9968     |
| cosine_precision@1  | 0.9737     |
| cosine_precision@3  | 0.3306     |
| cosine_precision@5  | 0.1987     |
| cosine_precision@10 | 0.0997     |
| cosine_recall@1     | 0.9737     |
| cosine_recall@3     | 0.9917     |
| cosine_recall@5     | 0.9936     |
| cosine_recall@10    | 0.9968     |
| cosine_ndcg@10      | 0.9865     |
| cosine_mrr@10       | 0.9831     |
| **cosine_map@100**  | **0.9832** |
| dot_accuracy@1      | 0.9737     |
| dot_accuracy@3      | 0.9917     |
| dot_accuracy@5      | 0.9936     |
| dot_accuracy@10     | 0.9968     |
| dot_precision@1     | 0.9737     |
| dot_precision@3     | 0.3306     |
| dot_precision@5     | 0.1987     |
| dot_precision@10    | 0.0997     |
| dot_recall@1        | 0.9737     |
| dot_recall@3        | 0.9917     |
| dot_recall@5        | 0.9936     |
| dot_recall@10       | 0.9968     |
| dot_ndcg@10         | 0.9865     |
| dot_mrr@10          | 0.9831     |
| dot_map@100         | 0.9832     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset
* Size: 303,863 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                               | positive                                                                             | negative                                                                              |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                               | string                                                                                |
  | details | <ul><li>min: 27 tokens</li><li>mean: 51.44 tokens</li><li>max: 137 tokens</li></ul> | <ul><li>min: 39 tokens</li><li>mean: 197.8 tokens</li><li>max: 1607 tokens</li></ul> | <ul><li>min: 48 tokens</li><li>mean: 224.41 tokens</li><li>max: 2735 tokens</li></ul> |
* Loss: [<code>CachedGISTEmbedLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedgistembedloss) with these parameters:

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 128
- `learning_rate`: 2e-05
- `num_train_epochs`: 1
- `warmup_ratio`: 0.1
- `fp16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Environmental Impact
Carbon emissions were measured using [CodeCarbon](https://github.com/mlco2/codecarbon).
- **Energy Consumed**: 5.517 kWh
- **Carbon Emitted**: 2.036 kg of CO2
- **Hours Used**: 9.954 hours

### Training Hardware
- **On Cloud**: No
- **GPU Model**: 1 x NVIDIA H100 NVL
- **CPU Model**: AMD EPYC 9V84 96-Core Processor
- **RAM Size**: 314.68 GB

### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.1.1
- Transformers: 4.44.2
- PyTorch: 2.3.0+cu121
- Accelerate: 0.33.0
- Datasets: 2.21.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2024,
  author =       {Louis Brulé Naudet},
  title =        {Lemone-Embed: A Series of Fine-Tuned Embedding Models for French Taxation},
  year =         {2024}
  howpublished = {\url{https://huggingface.co/datasets/louisbrulenaudet/lemone-embed-pro}},
}
```

## Feedback

If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).

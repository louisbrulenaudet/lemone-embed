# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib

from datetime import datetime
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Type,
    Tuple,
    Union,
    Mapping,
    TypeVar,
    Callable,
    Optional,
    Sequence,
)

import chromadb
import polars as pl

from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datasets import Dataset
from torch.cuda import is_available

client = chromadb.Client(
    settings=Settings(anonymized_telemetry=False)
)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="louisbrulenaudet/lemone-embed-pro",
    device="cuda" if is_available() else "cpu",
    trust_remote_code=True
)

collection = client.get_or_create_collection(
    name="tax",
    embedding_function=sentence_transformer_ef
)

bofip_dataframe = pl.scan_parquet(
    "hf://datasets/louisbrulenaudet/bofip/data/train-00000-of-00001.parquet"
).with_columns(
    [
        (
            pl.lit("Bulletin officiel des finances publiques - impôts").alias(
                "title_main"
            )
        ),
        (
            pl.col("debut_de_validite")
            .str.strptime(pl.Date, format="%Y-%m-%d")
            .dt.strftime("%Y-%m-%d 00:00:00")
        ).alias("date_publication"),
        (
            pl.col("contenu")
            .map_elements(lambda x: hashlib.sha256(str(x).encode()).hexdigest(), return_dtype=pl.Utf8)
            .alias("hash")
        )
    ]
).rename(
    {
        "contenu": "text",
        "permalien": "url_sourcepage",
        "identifiant_juridique": "id_sub",
    }
).select(
    [
        "text",
        "title_main",
        "id_sub",
        "url_sourcepage",
        "date_publication",
        "hash"
    ]
)

books: List[str] = [
    "hf://datasets/louisbrulenaudet/code-douanes/data/train-00000-of-00001.parquet",
    "hf://datasets/louisbrulenaudet/code-impots/data/train-00000-of-00001.parquet",
    "hf://datasets/louisbrulenaudet/code-impots-annexe-i/data/train-00000-of-00001.parquet",
    "hf://datasets/louisbrulenaudet/code-impots-annexe-ii/data/train-00000-of-00001.parquet",
    "hf://datasets/louisbrulenaudet/code-impots-annexe-iii/data/train-00000-of-00001.parquet",
    "hf://datasets/louisbrulenaudet/code-impots-annexe-iv/data/train-00000-of-00001.parquet",
    "hf://datasets/louisbrulenaudet/code-impositions-biens-services/data/train-00000-of-00001.parquet",
    "hf://datasets/louisbrulenaudet/livre-procedures-fiscales/data/train-00000-of-00001.parquet"
]

legi_dataframe = pl.concat(
    [
        pl.scan_parquet(
            book
        ) for book in books
    ]
).with_columns(
    [
        (
            pl.lit("https://www.legifrance.gouv.fr/codes/article_lc/")
            .add(pl.col("id"))
            .alias("url_sourcepage")
        ),
        (
            pl.col("dateDebut")
            .cast(pl.Int64)
            .map_elements(
                lambda x: datetime.fromtimestamp(x / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                return_dtype=pl.Utf8
            )
            .alias("date_publication")
        ),
        (
            pl.col("texte")
            .map_elements(lambda x: hashlib.sha256(str(x).encode()).hexdigest(), return_dtype=pl.Utf8)
            .alias("hash")
        )
    ]
).rename(
    {
        "texte": "text",
        "num": "id_sub",
    }
).select(
    [
        "text",
        "title_main",
        "id_sub",
        "url_sourcepage",
        "date_publication",
        "hash"
    ]
)

print("Starting embeddings production...")

dataframe = pl.concat(
    [
        bofip_dataframe,
        legi_dataframe
    ]
).filter(
    pl.col(
        "text"
    ).is_not_null()
).with_columns(
    pl.col("text").map_elements(
        lambda x: sentence_transformer_ef(
            [x]
        )[0].tolist(),
        return_dtype=pl.List(pl.Float64)
    ).alias("lemone_pro_embeddings")
).collect()

dataset = Dataset.from_pandas(dataframe.to_pandas())

dataset.push_to_hub(
    "louisbrulenaudet/lemone-docs-embeded",
    token="hf_"
)
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

dataframe = pl.scan_parquet(
  "hf://datasets/louisbrulenaudet/lemone-docs-embeded/data/train-00000-of-00001.parquet"
).filter(
    pl.col(
        "text"
    ).is_not_null()
).collect()

collection.add(
    embeddings=dataframe["lemone_pro_embeddings"].to_list(),
    documents=dataframe["text"].to_list(),
    metadatas=dataframe.remove_columns(
        [
            "lemone_pro_embeddings", 
            "text"
        ]
    ).to_list(),
    ids=[
        str(i) for i in range(0, dataframe.shape[0])
    ]
)

collection.query(
    query_texts=["Les personnes morales de droit public ne sont pas assujetties à la taxe sur la valeur ajoutée pour l'activité de leurs services administratifs, sociaux, éducatifs, culturels et sportifs lorsque leur non-assujettissement n'entraîne pas de distorsions dans les conditions de la concurrence."],
    n_results=10,
)
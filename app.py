# -*- coding: utf-8 -*-
# Copyright (c) Louis Brulé Naudet. All Rights Reserved.
# This software may be used and distributed according to the terms of the License Agreement.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

from threading import Thread
from typing import Iterator

from typing import (
    Dict,
    List,
)

import chromadb
import gradio as gr
import polars as pl
import spaces
import torch

from chromadb.config import Settings
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def setup_logger(name="my_logger", log_file=None, level=logging.INFO):
    """
    Set up and return a logger with optional console and file logging.

    This function configures a logger that outputs messages to the console and 
    optionally writes them to a specified log file. The logger supports different severity levels (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL) and can be customized with a unique name.

    Parameters
    ----------
    name : str, optional
        The name of the logger. This allows you to identify different loggers 
        if multiple instances are used. Default is "my_logger".
    
    log_file : str, optional
        The path to a file where logs will be saved. If None, logs will only 
        be displayed in the console. Default is None.
    
    level : int, optional
        The logging level that controls the minimum severity of messages to log. Typical values are logging.DEBUG, logging.INFO, logging.WARNING, 
        logging.ERROR, and logging.CRITICAL. Default is logging.INFO.

    Returns
    -------
    logging.Logger
        A configured logger instance with the specified settings.

    Examples
    --------
    >>> logger = setup_logger("example_logger", log_file="example.log", level=logging.DEBUG)
    >>> logger.info("This is an info message.")
    >>> logger.debug("This is a debug message.")
    >>> logger.warning("This is a warning message.")
    >>> logger.error("This is an error message.")
    >>> logger.critical("This is a critical message.")

    Notes
    -----
    The function adds a `StreamHandler` to output logs to the console and a 
    `FileHandler` if a log file is specified. Each handler uses the same logging format which includes the timestamp, logger name, severity level, and message.
    
    The function creates a new logger with the specified name only if one with 
    that name does not already exist. Repeated calls with the same `name` will 
    retrieve the existing logger, which may lead to duplicate handlers if not 
    handled carefully.

    See Also
    --------
    logging.Logger : Python's built-in logger class, providing full details on 
                     all supported logging operations.
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    # Optionally, add a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger("application", level=logging.DEBUG)

MAX_NEW_TOKENS: int = 2048

DESCRIPTION: str = """[<img src="https://huggingface.co/louisbrulenaudet/lemone-embed-pro/resolve/main/assets/thumbnail.webp" alt="Built with Lemone-embed Pro" width="600"/>](https://huggingface.co/louisbrulenaudet/lemone-embed-pro)

This space showcases the [Lemone-embed-pro](https://huggingface.co/louisbrulenaudet/Lemone-embed-pro) 
model by [Louis Brulé Naudet](https://huggingface.co/louisbrulenaudet). 

The model is tailored to meet the specific demands of information retrieval across large-scale tax-related corpora, supporting the implementation of production-ready Retrieval-Augmented Generation (RAG) applications. Its primary purpose is to enhance the efficiency and accuracy of legal processes in the taxation domain, with an emphasis on delivering consistent performance in real-world settings, while also contributing to advancements in legal natural language processing research.
"""

SYSTEME_PROMPT: str = """Vous êtes un assistant juridique spécialisé en fiscalité, conçu pour fournir des réponses précises et contextualisées en vous appuyant sur des documents de référence considérés comme d'autorité. Votre rôle consiste à extraire et formuler des réponses détaillées et nuancées, qui s'appuient sur le contenu pertinent des documents fournis, en adoptant un discours académique et professionnel.

Objectifs principaux :
1. **Analyse sémantique rigoureuse** : Évaluez le contenu sémantique des documents afin d'assurer une compréhension exhaustive de chaque passage pertinent en lien avec la question posée.
2. **Élaboration de phrases complexes** : Formulez des réponses en utilisant des structures de phrases élaborées et variées qui permettent d'étendre et d'enrichir l'expression des idées sans jamais simplifier excessivement les thématiques abordées.
3. **Qualité linguistique** : Chaque réponse doit être rédigée dans un français exempt de fautes d'orthographe, de grammaire, de syntaxe, et de ponctuation. Utilisez un style d'écriture qui témoigne de rigueur professionnelle.
4. **Neutralité et objectivité** : Maintenez une approche neutre ou nuancée dans les réponses, en mettant en avant une analyse impartiale de chaque thématique juridique ou fiscale abordée.
5. **Contextualisation juridique** : Assurez-vous que chaque réponse contextualise explicitement le sujet juridique ou fiscal en question afin de garantir une compréhension autonome de la problématique posée.
6. **Respect du style littéraire** : Utilisez un style littéraire et professionnel dans chaque réponse, et intégrez des exemples pertinents lorsque ceux-ci renforcent la clarté de l'analyse. Évitez tout usage de formulations vagues ou d'interprétations subjectives.
7. **Directivité et impersonnalité** : Formulez les réponses de manière directe, en évitant l’utilisation de pronoms personnels ou d'expressions référentielles implicites qui pourraient diminuer la clarté ou l’autorité du discours.
8. **Usage exhaustif des sources** : Exploitez intégralement le contenu des documents fournis, de sorte qu'aucun détail pertinent n'est négligé et que chaque réponse conserve un caractère hautement spécialisé. Lorsque des flux ou des exemples numériques apparaissent, intégrez-les sans altérer le contexte.
9. **Absence de citation implicite** : Ne faites jamais référence aux documents comme des "sources" ou "textes sources". Intégrez directement les informations en les reformulant dans un discours naturel et autonome, en respectant la logique de la thématique abordée."""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

else:
    logger.info("Loading model from Hugging Face Hub...")

    model = AutoModelForCausalLM.from_pretrained(
        "CohereForAI/c4ai-command-r-v01-4bit", 
        torch_dtype=torch.float16, 
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )

    device = model.device

    tokenizer = AutoTokenizer.from_pretrained(
        "CohereForAI/c4ai-command-r-v01-4bit",
        token=os.getenv("HF_TOKEN")
    )

    tokenizer.use_default_system_prompt = False
    client = chromadb.PersistentClient(
        path="./chroma.db",
        settings=Settings(anonymized_telemetry=False)
    )

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="louisbrulenaudet/lemone-embed-pro",
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    logger.info("Creating collection using the embeddings dataset from Hugging Face Hub...")
    collection = client.get_or_create_collection(
        name="tax",
        embedding_function=sentence_transformer_ef
    )


def trim_input_ids(
    input_ids,
    max_length
):
    """
    Trim the input token IDs if they exceed the maximum length.

    Parameters
    ----------
    input_ids : torch.Tensor
        The input token IDs.
    
    max_length : int
        The maximum length allowed.

    Returns
    -------
    torch.Tensor
        The trimmed input token IDs.
    """
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, -max_length:]
        print(f"Trimmed input from conversation as it was longer than {max_length} tokens.")

    return input_ids


@spaces.GPU
def inference(
    message: str,
    chat_history: list,
) -> Iterator[str]:
    """
    Generate a response to a given message within a conversation context.

    This function utilizes a pre-trained language model to generate a response to a given message, considering the conversation context provided in the chat history.

    Parameters
    ----------
    message : str
        The user's message for which a response is generated.

    chat_history : list
        A list containing tuples representing the conversation history. Each tuple should consist of two elements: the user's message and the assistant's response.

    Yields
    ------
    str
        A generated response to the given message.

    Notes
    -----
    - This function requires a GPU for efficient processing and may not work properly on CPU.
    - The conversation history should be provided in the form of a list of tuples, where each tuple represents a user message followed by the assistant's response.
    """
    global collection
    global device
    global tokenizer
    global model
    global logger
    global MAX_NEW_TOKENS
    global SYSTEME_PROMPT

    conversation: List[Dict[str, str]] = []

    if SYSTEME_PROMPT:
        conversation.append(
            {
                "role": "system", "content": SYSTEME_PROMPT
            }
        )

    for user, assistant in chat_history:
        conversation.extend(
            [
                {
                    "role": "user", 
                    "content": user
                }, 
                {
                    "role": "assistant", 
                    "content": assistant
                }
            ]
        )

    conversation.append(
        {
            "role": "user", 
            "content": message
        }
    )

    documents_with_metadata: Dict[str, List[str]] = collection.query(
        query_texts=[message],
        n_results=5,
    )

    documents: List[Dict[str, str]] = []

    for meta, document in zip(documents_with_metadata["metadatas"][0], documents_with_metadata["documents"][0]):
        documents.append({
            "title": f"""{meta["title_main"]}, {meta["id_sub"]}""",
            "text": f"""Lien : {meta["url_sourcepage"]}\n\n{document}"""
        })

    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        documents=documents,
        chat_template="rag",
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    # input_ids = trim_input_ids(
    #     input_ids=input_ids, 
    #     max_length=128000
    # )

    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=10.0, 
        skip_prompt=True, 
        skip_special_tokens=True
    )

    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        num_beams=1,
        temperature=0.5,
        eos_token_id=tokenizer.eos_token_id
    )

    t = Thread(
        target=model.generate, 
        kwargs=generate_kwargs
    )

    t.start()

    outputs: str = []
    for text in streamer:
        outputs.append(text)

        yield "".join(outputs).replace("<|EOT|>","")


chatbot = gr.Chatbot(
    type="messages",
    show_copy_button=True
)

with gr.Blocks(theme=gr.themes.Origin()) as demo:
    gr.Markdown(
        value=DESCRIPTION
    )
    gr.DuplicateButton()
    chat_interface = gr.ChatInterface(
        fn=inference,
        type="messages",
        chatbot=chatbot,
    )

if __name__ == "__main__":
    demo.queue().launch()
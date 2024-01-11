import logging
import re
import time
import unicodedata

from io import BytesIO
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import tiktoken
from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.config import Settings
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pandas import DataFrame
from pypdf import PdfReader
from spacy.lang.en import English

from db import Chat, DATA_DIR

logging.basicConfig()
logging.getLogger().setLevel(getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME: str = 'text-embedding-ada-002' # max tokens = 8191 (as of 12/11/23)
LLM_NAME: str = 'gpt-3.5-turbo' # max tokens = 16385 (as of 12/11/23)
LLM_CLIENT: OpenAI = OpenAI(api_key='')
LLM_TEMPERATURE: int = 0
LLM_MAX_TOKENS: int = 150
VECTOR_DB_PATH: Path = DATA_DIR / 'vectorstore'
VECTOR_DB_CLIENT: ClientAPI = PersistentClient(path=str(VECTOR_DB_PATH), settings=Settings(allow_reset=True, anonymized_telemetry=False))
MAX_TOKENS_PER_CHUNK: int = 100
SYSTEM_PROMPT = 'You are a helpful AI bot that a user can chat with. You answer questions for the user based on your knowledge supplemented with any context given before the question. You may ask the user clarifying questions if needed to understand the question, or simply respond "I don\'t know" if you don\'t have an accurate answer. Do not mention that a context was provided, just try to use it to inform your responses.'
PROMPT = 'Context: {}\n\n---\n\nQuestion: {}\nAnswer:'

def process_pdf_bytes(file_bytes: BytesIO, filename: str):
    """
    Process bytes from uploaded pdf file: extract text, chunk
    text, embed text, save embedings to vectorstore
    """
    df = get_text_from_pdf(file_bytes, filename)
    df = count_tokens_in_text(df)
    # # check token histogram
    # df.token_count.hist()
    # plt.show()
    df_chunked = chunk_text(df, MAX_TOKENS_PER_CHUNK)
    # # check token histogram again
    # df_chunked = count_tokens_in_text(df_chunked)
    # df_chunked.token_count.hist()
    # plt.show()
    df_chunked = embed_text(df_chunked)
    save_to_vectorstore(df_chunked)


def get_text_from_pdf(file_bytes: BytesIO, filename: str) -> DataFrame:
    """
    Extract text from pdf and return a DataFrame with columns:
    filename, page, text
    """
    # read text from pdf into a dataframe
    pdf: PdfReader = PdfReader(file_bytes)
    records = []
    for i, page in enumerate(pdf.pages):
        records.append({'filename': filename, 'page': i, 'text': page.extract_text()})

    df = DataFrame.from_records(records)
    # clean whitespace
    df.text = df.text.apply(clean_whitespace)

    return df


def count_tokens_in_text(df: DataFrame) -> DataFrame:
    """
    Count the tokens in each row of the `text` column and save
    the counts in a new `token_count` column
    Return same dataframe with new column added
    """
    # cl100k_base tokenizer is designed to work with the text-embedding-ada-002 model
    tokenizer = tiktoken.get_encoding('cl100k_base')
    df['token_count'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    return df


def chunk_text(df: DataFrame, max_tokens: int) -> DataFrame:
    """
    Parse all text from `text` column into sentences
    Create chunks of text with less than `max_tokens` tokens
    Return new dataframe with chunked text with the same columns:
    filename, page, text
    """

    tokenizer = tiktoken.get_encoding('cl100k_base')

    # [(filename, page, sentence list, tokens list)]
    sentence_data: List[Tuple[str, int, List[str], List[int]]] = []
    # iterate through each row / pdf page (simpler to concatenate all text, but want to preserve page number metadata)
    for row in df.iterrows():
        # parse into sentences
        nlp = English()
        nlp.add_pipe('sentencizer')
        doc_spacy = nlp(row[1].text)
        curr_sentences = [sent.text.strip() for sent in doc_spacy.sents]
        # count tokens by sentence
        num_tokens_by_sentence = [len(tokenizer.encode(" " + s)) for s in curr_sentences]
        sentence_data.append((row[1].filename, row[1].page, curr_sentences, num_tokens_by_sentence))

    # create chunks of text for embedding with less than max_tokens
    chunks: List[Dict[str, str | int]] = []
    curr_chunk: List[str] = []
    curr_chunk_tokens: int = 0

    for filename, page, sentences, num_tokens in sentence_data:
        for curr_sentence, curr_num_tokens in zip(sentences, num_tokens):
            # skip sentences longer than the max token length
            if curr_num_tokens > max_tokens:
                continue
            # if this sentence puts us over the max tokens, add it along with
            # the current chunk to the list of chunks and reset
            if curr_chunk_tokens + curr_num_tokens > max_tokens:
                chunks.append({'text': ' '.join(curr_chunk), 'page': page, 'filename': filename})
                curr_chunk = []
                curr_chunk_tokens = 0
            # add sentence to current chunk and track tokens
            curr_chunk.append(curr_sentence)
            curr_chunk_tokens += curr_num_tokens

    df_chunked = DataFrame.from_records(chunks)
    return df_chunked


def embed_text(df: DataFrame, save_csv=False) -> DataFrame:
    """
    Embed the text in each row of the `text` column and save
    the embeddings in a new `embedding` column
    Optionally save the dataframe to a csv file
    Return same dataframe with new column added
    """
    df['embedding'] = df.text.apply(lambda x: get_embedding(x))
    if save_csv:
        df.to_csv(f'embeddings_{int(time.time())}.csv')
    return df


def get_embedding(text: str) -> List[float | int]:
    """
    Get embedding for single chunk of text and return it as a list
    """
    response: CreateEmbeddingResponse = LLM_CLIENT.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL_NAME
    )
    logger.info(f'{response.usage.prompt_tokens=}')
    logger.info(f'{response.usage.total_tokens=}')
    # logger.debug(f'{dict(response)}')
    return response.data[0].embedding


def save_to_vectorstore(df: DataFrame) -> None:
    # add vectorstore metadata
    df['chroma_metadata'] = df.page.apply(lambda x: {'page': x})
    collection = VECTOR_DB_CLIENT.get_or_create_collection(name='docs-chat')
    collection.add(
        documents=list(df.text),
        embeddings=list(df.embedding),
        metadatas=list(df.chroma_metadata),
        ids=[str(uuid4()) for _ in range(len(df))]
    )


def get_document_context(user_message) -> List[str]:
    collection = VECTOR_DB_CLIENT.get_or_create_collection(name='docs-chat')
    results = collection.query(
        query_embeddings=get_embedding(user_message),
        n_results=5,
        include=['documents', 'metadatas', 'distances']
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains":"search_string"}
    )
    if not results:
        return []
    # distances = [d for d in results['distances'][0]]
    # maybe check the distances for a threshold?
    documents = [d for d in results['documents'][0]]
    return documents


def get_bot_response(user_message: str, document_context: List[str], chat: Chat) -> str:
    prev_messages: List[ChatCompletionMessageParam] = []
    for m in chat.messages:
        mdict: ChatCompletionMessageParam = {'role': 'user', 'content': m.text}
        prev_messages.append(mdict)
    current_prompt = PROMPT.format(document_context, user_message)
    logger.debug(current_prompt)
    messages: List[ChatCompletionMessageParam] = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        *prev_messages,
        {'role': 'user', 'content': current_prompt}
    ]
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_NAME,
            messages = messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return str(response.choices[0].message.content)
    except Exception as e:
        logger.exception('LLM call problem')
        return ''


def save_to_file(file_bytes: BytesIO, filename: str, filedir: Path):
    with open(filedir / filename, 'wb') as f:
        f.write(file_bytes.read())


def slugify(value: Any) -> str:
    """
    Based off Django Framework code - https://github.com/django/django/blob/main/django/utils/text.py
    """
    value = str(value)
    value = (
        unicodedata.normalize('NFKD', value)
        .encode('ascii', 'ignore')
        .decode('ascii')
    )
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def clean_whitespace(text):
    return re.sub(r'\s+', r' ', text)


def check_api_key() -> None:
    models = LLM_CLIENT.models.list()
    logger.debug(models)


def set_api_key(api_key: str) -> None:
    LLM_CLIENT.api_key = api_key

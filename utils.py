import logging
import re
import unicodedata

from io import BytesIO
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import tiktoken
from chromadb import PersistentClient
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
VECTOR_DB_CLIENT = PersistentClient(path=str(VECTOR_DB_PATH), settings=Settings(anonymized_telemetry=False))



def check_api_key() -> None:
    models = LLM_CLIENT.models.list()
    logger.debug(models)


def set_api_key(api_key: str) -> None:
    LLM_CLIENT.api_key = api_key


def get_embedding(text: str) -> List[float | int]:
    response: CreateEmbeddingResponse = LLM_CLIENT.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL_NAME
    )
    logger.info(f'{response.usage.prompt_tokens=}')
    logger.info(f'{response.usage.total_tokens=}')
    logger.debug(f'{dict(response)}')
    return response.data[0].embedding


def clean_whitespace(text):
    return re.sub(r'\s+', r' ', text)


def save_to_vectorstore(file_bytes: BytesIO, filename: str):
    # read text from pdf into a dataframe
    pdf: PdfReader = PdfReader(file_bytes)
    records = []
    for i, page in enumerate(pdf.pages):
        records.append({'page': i, 'filename': filename, 'text': page.extract_text()})

    df = DataFrame.from_records(records)
    # clean whitespace
    df.text = df.text.apply(clean_whitespace)

    # cl100k_base tokenizer is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding('cl100k_base')
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # df.n_tokens.hist()
    # plt.show()

    sentence_data: List[Tuple[List[str], List[int], int]] = [] # (sentence list, num_tokens_by_sentence, page)
    # iterate through each row / pdf page (simpler to concatenate all text, but want to preserve page number metadata)
    for row in df.iterrows():
        # parse into sentences
        nlp = English()
        nlp.add_pipe('sentencizer')
        doc_spacy = nlp(row[1].text)
        curr_sentences = [sent.text.strip() for sent in doc_spacy.sents]
        # count tokens by sentence
        num_tokens_by_sentence = [len(tokenizer.encode(" " + s)) for s in curr_sentences]
        sentence_data.append((curr_sentences, num_tokens_by_sentence, row[1].page))

    # create chunks of text for embedding with less than MAX_TOKENS
    MAX_TOKENS: int = 256
    chunks: List[Dict[str, str | int]] = []
    curr_chunk: List[str] = []
    curr_chunk_tokens: int = 0

    for sentences, num_tokens, page in sentence_data:
        for curr_sentence, curr_num_tokens in zip(sentences, num_tokens):
            # skip sentences longer than the max token length
            if curr_num_tokens > MAX_TOKENS:
                continue
            # if this sentence puts us over the max tokens, add it along with
            # the current chunk to the list of chunks and reset
            if curr_chunk_tokens + curr_num_tokens > MAX_TOKENS:
                chunks.append({'text': ' '.join(curr_chunk), 'page': page, 'filename': filename})
                curr_chunk = []
                curr_chunk_tokens = 0
            # add sentence to current chunk and track tokens
            curr_chunk.append(curr_sentence)
            curr_chunk_tokens += curr_num_tokens

    df_chunked = DataFrame.from_records(chunks)

    # check token histogram again
    # df_chunked['n_tokens'] = df_chunked.text.apply(lambda x: len(tokenizer.encode(x)))
    # df_chunked.n_tokens.hist()
    # plt.show()

    # embed each chunk
    df_chunked['embedding'] = df_chunked.text.apply(lambda x: get_embedding(x))
    # df_chunked.to_csv('embeddings.csv')

    # add metadata
    df_chunked['chroma_metadata'] = df_chunked.page.apply(lambda x: {'page': x})

    collection = VECTOR_DB_CLIENT.get_or_create_collection(name='docs-chat')
    collection.add(
        documents=list(df_chunked.text),
        embeddings=list(df_chunked.embedding),
        metadatas=list(df_chunked.chroma_metadata),
        ids=[str(uuid4()) for _ in range(len(df_chunked))]
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

    # PROMPTS:
    # Given the following extracted parts of documents from a database and a question,
    # create a final answer with references ("SOURCES").
    # If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    # ALWAYS return a "SOURCES" part in your answer.

    # Answer the question based only on the following context:
    # {context}
    # Question: {question}

    # "Answer the question based on the context below,
    # and if the question can't be answered based on the context, say \"I don't know\"\n\n"
    # "Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"
    prev_messages: List[ChatCompletionMessageParam] = []
    for m in chat.messages:
        mdict: ChatCompletionMessageParam = {'role': 'user', 'content': m.text}
        prev_messages.append(mdict)
    prompt = f'Context: {document_context}\n\n---\n\nQuestion: {user_message}\nAnswer:'
    logger.debug(prompt)
    messages: List[ChatCompletionMessageParam] = [
        {'role': 'system', 'content': 'You are a helpful AI bot that a user can chat with. You answer questions for the user based on any context given before the question. You may ask the user clarifying questions if needed to understand the question, or simply respond "I don\'t know" if you don\'t have an accurate answer. You shouldn\'t say "based on the provided context" or similar phrases, since the user knows you have the context for the answers.'},
        *prev_messages,
        {'role': 'user', 'content': prompt}
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


# chunk_size = 3
# overlap = 2
# def chunk(sequence, chunk_size, overlap):
#     """
#     Split a sequence into chunks of given size with given overlap
#     """
#     result = []
#     for i in range(0, len(sequence) - overlap, chunk_size - overlap):
#         result.append(sequence[i:i + chunk_size])
#     return result

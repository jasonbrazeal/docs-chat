import logging
from contextlib import asynccontextmanager
from io import BytesIO
from os import getenv
from pathlib import Path
from shutil import rmtree
from typing import Annotated, Optional

from chromadb.config import Settings
from fastapi import FastAPI, Request, Header, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from httpx import AsyncClient
from openai import OpenAI
from pypdf import PdfReader
from sqlmodel import Session, select
from sqlalchemy.exc import NoResultFound

from db import create_db_engine, Message, Sender, Chat, ApiKey, PdfDocument

logging.basicConfig()
logging.getLogger().setLevel(getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

DOCS_PATH = Path(__file__).parent / 'vectorstore'
DB_PATH = Path(__file__).parent / 'chat.db'
MAX_FILESIZE_MB = 100

db_engine = create_db_engine(DB_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncClient()
    EMBEDDINGS = None
    LLM = None
    with Session(db_engine) as session:
        statement = select(ApiKey)
        results = session.exec(statement)
        api_keys = list(results)
        if api_keys:
            api_key = api_keys[0]
            LLM = OpenAI(openai_api_key=api_key.cred, model_name='gpt-3.5-turbo')
        else:
            logger.info('No API key found in db')
    yield
    await client.aclose()

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')
async def index(request: Request, hx_request: Optional[str] = Header(None)):
    context = {'request': request}
    return templates.TemplateResponse('index.html', context)

@app.post('/api-key')
async def api_key(request: Request, hx_request: Optional[str] = Header(None)):
    form = await request.form()
    api_key = form.get('api-key-input', '')
    if hx_request and api_key:
        with Session(db_engine) as session:
            statement = select(ApiKey)
            results = session.exec(statement)
            try:
                existing_api_key = results.one()
                session.delete(existing_api_key)
            except NoResultFound:
                pass
            api_key_obj = ApiKey(cred=api_key)
            session.add(api_key_obj)
            session.commit()
            global EMBEDDINGS
            global LLM
            # TODO: error handling for incorrect api key
            LLM = OpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo')
            api_key_masked = (len(api_key) - 4) * '*' + api_key[-4:]
            return PlainTextResponse(api_key_masked)
    return HTMLResponse('<button data-target="api-key-modal" class="btn modal-trigger green darken-4">Set OpenAI API key</button>')

@app.get('/settings')
async def settings(request: Request, hx_request: Optional[str] = Header(None)):
    with Session(db_engine) as session:
        statement = select(ApiKey)
        results = session.exec(statement)
        api_keys = list(results)
        try:
            api_key = api_keys[0]
        except IndexError:
            api_key = None
    api_key_masked = ''
    if api_key is not None:
        api_key_masked = (len(api_key.cred) - 4) * '*' + api_key.cred[-4:]
    context = {'request': request, 'api_key_masked': api_key_masked, 'js_file': 'settings.js'}
    return templates.TemplateResponse('settings.html', context)

@app.get('/history')
async def history(request: Request, hx_request: Optional[str] = Header(None)):
    with Session(db_engine) as session:
        statement = select(Chat).order_by(Chat.created_at.desc())
        results = session.exec(statement)
        chats = list(results)
        [chat.messages for chat in chats] # load messages for all chats
    context = {'request': request, 'chats': chats, 'js_file': 'history.js'}
    return templates.TemplateResponse('history.html', context)

@app.get('/chat/new')
async def new_chat(request: Request, hx_request: Optional[str] = Header(None)):
    with Session(db_engine) as session:
        chat = Chat()
        session.add(chat)
        session.commit()
    return RedirectResponse('/chat', status_code=302)

@app.get('/chat/{chat_id}')
async def chat_data(request: Request, chat_id: Annotated[int, Path()], hx_request: Optional[str] = Header(None)):
    with Session(db_engine) as session:
        chat = session.get(Chat, chat_id)
        chat_data = ''
        for message in chat.messages:
            chat_data += f'<p>{message.sender}: {message.text}</p>'
    return HTMLResponse(chat_data)

@app.get('/chat')
@app.post('/chat')
async def chat(request: Request, hx_request: Optional[str] = Header(None)):
    context = {'request': request, 'js_file': 'chat.js'}
    # load last conversation if one is present
    with Session(db_engine) as session:
        statement = select(Chat).order_by(Chat.created_at.desc()).limit(1)
        results = session.exec(statement)
        chats = list(results)
        if chats:
            chat = chats[0]
            chat.messages = list(reversed(chat.messages))
            context['chat'] = chat
        else:
            return RedirectResponse('/chat/new', status_code=302)

        if hx_request:
            form = await request.form()
            user_message = form.get('user_message', '').strip()
            if not user_message:
                return HTMLResponse('')
            vectordb = Chroma(
                embedding_function=EMBEDDINGS,
                persist_directory=str(DOCS_PATH),
                client_settings=Settings(anonymized_telemetry=False)
            )
            retriever = vectordb.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=LLM, chain_type='stuff', retriever=retriever)
            bot_message = qa.run(user_message)
            context['messages'] = [Message(text=user_message, sender=Sender.USER.name, chat_id=chat.id),
                                   Message(text=bot_message, sender=Sender.BOT.name, chat_id=chat.id)]
            for message in context['messages']:
                session.add(message)
            session.commit()
            session.refresh(chat)
            chat.messages = list(reversed(chat.messages))
            return templates.TemplateResponse('chat_table.html', context)

    return templates.TemplateResponse('chat.html', context)

@app.get('/documents')
async def documents(request: Request, hx_request: Optional[str] = Header(None)):
    with Session(db_engine) as session:
        statement = select(PdfDocument)
        results = session.exec(statement)
        documents = list(results)
    context = {'request': request, 'documents': documents, 'js_file': 'documents.js'}
    return templates.TemplateResponse('documents.html', context)

@app.post('/upload')
async def upload(request: Request, documents: list[UploadFile], hx_request: Optional[str] = Header(None)):
    all_document_objs = []
    total_bytes = 0
    with Session(db_engine) as session:
        for document in documents:
            file_bytes = BytesIO(await document.read())
            num_file_bytes = len(file_bytes.read())
            num_file_megabytes = round(float(num_file_bytes) / 1000000, 2)
            file_bytes.seek(0)

            if num_file_megabytes > MAX_FILESIZE_MB:
                continue

            doc = PdfDocument(filename=document.filename)
            session.add(doc)
            session.commit()

            pdf_reader = PdfReader(file_bytes)
            document_objs = [
                Document(page_content=page.extract_text(),
                         metadata={'source': doc.filename, 'page': i})
                for i, page in enumerate(pdf_reader.pages)
            ]
            all_document_objs.extend(document_objs)
    if not all_document_objs:
        return RedirectResponse('/documents', status_code=302)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = splitter.split_documents(all_document_objs)

    vectordb = Chroma.from_documents(documents=docs_split,
                                     embedding=EMBEDDINGS,
                                     persist_directory=str(DOCS_PATH))
    # save vectorstore to disk
    vectordb.persist()

    return RedirectResponse('/documents', status_code=302)

@app.post('/clear')
async def clear(request: Request, hx_request: Optional[str] = Header(None)):
    with Session(db_engine) as session:
        statement = select(PdfDocument)
        results = session.exec(statement)
        for doc in list(results):
            session.delete(doc)
        session.commit()
    rmtree(str(DOCS_PATH), ignore_errors=True)
    DOCS_PATH.mkdir()
    return RedirectResponse(request.url_for('documents'), status_code=302)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)

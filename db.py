import logging

from datetime import datetime
from enum import auto, Enum
from pathlib import Path
from typing import List, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, Relationship


class Sender(Enum):
    USER = auto()
    BOT = auto()


class Chat(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    messages: List['Message'] = Relationship(back_populates='chat')
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    sender: str
    chat_id: Optional[int] = Field(default=None, foreign_key='chat.id')
    chat: Optional[Chat] = Relationship(back_populates='messages')
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class ApiKey(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    cred: str
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class PdfDocument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


def create_db_engine(db_path: str):
    db_url = f'sqlite:///{db_path}'
    engine = create_engine(
        db_url, echo=logging.getLogger().isEnabledFor(logging.DEBUG),
        connect_args={'check_same_thread': db_url.startswith('sqlite')},
    )
    return engine


def create_db(engine):
    SQLModel.metadata.create_all(engine)


def init_db(engine, num_chats=2):
    messages1 = [
        'Hi there! Would you like to chat about your documents?',
        'Sure',
        "OK, feel free to ask anything you'd like about the knowledge base",
        'What is a metaclass in Python?'
    ]
    messages2 = [
        'Hi there! Would you like to chat about your documents?',
        'yes please',
        "OK, feel free to ask anything you'd like about the pdfs you uploaded",
        'Alright, can you summarize the paper I uploaded?'
    ]
    documents = ['Reasoning and Acting: An LLM Reasoning Framework', 'Survey of Augmented Language Models']

    with Session(engine) as session:
        for i in range(num_chats):
            chat = Chat()
            session.add(chat)
            session.commit()
            if i % 2 == 0:
                messages = messages1
            else:
                messages = messages2
            for i, message in enumerate(messages):
                sender = Sender.BOT if i % 2 == 0 else Sender.USER
                session.add(Message(chat_id=chat.id, text=message, sender=sender.name))
        session.commit()
        for name in documents:
            doc = PdfDocument(filename=name)
            session.add(doc)
        session.commit()

if __name__ == '__main__':
    DB_PATH = Path('./chat.db')
    DB_PATH.unlink(missing_ok=True)
    db_engine = create_db_engine(str(DB_PATH))
    create_db(db_engine)
    # init_db(db_engine)

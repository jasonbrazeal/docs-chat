# from pathlib import Path

# from httpx import AsyncClient


# async def main():
#     client = AsyncClient()

#     # download htmx
#     # download normalize.css
#     request = client.build_request('GET', 'https://necolas.github.io/normalize.css/8.0.1/normalize.css')
#     response = await client.send(request)
#     breakpoint()
#     print(response)


# if __name__ == '__main__':
#     main()


from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from openai import Completion, ChatCompletion

# intial load
loader = PyPDFLoader('docs/dsp.pdf')
docs = loader.load_and_split()
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory='/home/jason/docs-chat/docs/vectorstore')
# save
vectordb.persist()


# reload
vectordb = Chroma(embedding_function=embedding, persist_directory='/home/jason/docs-chat/docs/vectorstore')


# semantic similarity search
query = 'what is dsp?'
docs = vectordb.similarity_search(query)

# retriever
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=retriever)
query = 'what is dsp?'
qa.run(query)

# completions API (no vectorstore used)
completion_response = Completion.create(
  model='text-davinci-003',
  prompt='What is DSP?'
)

# chat completions API (single message, no vectorstore used)
chat_completion_response = ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
        {'role': 'user', 'content': 'What is DSP?'}
    ]
)

# chat completions API can include chat messages for context, e.g.
# messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
# ]

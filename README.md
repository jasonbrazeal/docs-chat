# Docs Chat

Docs Chat is a small application I created to have a demo of the kind of things I have been working on lately. It is not a production-ready application; there are no tests and no optimizations on the LLM or RAG side of things. Out of the box, it is essentially a wrapper around an LLM. But once you upload your own documents, the LLM then gets the relevant context from your documents which enables it to assist in a much more productive way. These documents could be proprietary business information or data, more up-to-date news articles (GPT-4 Turbo's cutoff date is April 2023), or anything else you want the AI to be able to discuss. Again, I do not intend for Docs Chat to be used in a production environment, but rather as a starting point for discussion about retrieval-augmented generation applications.

## How It Works

It's an asynchronous Python web application (FastAPI, SQLModel) that uses OpenAI's Completions API for the response generation. You have to provide an API key before chatting, and a history of all your chats is maintained in the database. There is a page that allows pdf documents to be uploaded, and doing so kicks off a job that processes the documents and saves the embedded text into a vector database, Chroma. When the user sends a message, Chroma is queried and if text is present that is relevant to the user's message, it is included in the LLM prompt as part of the context. Having this context allows the LLM to provide more up-to-date and relevant responses to the user's queries.

## Improvements

If this application was developed for a real business use case, there would be specific requirements and a plan of work with milestones that I would be working toward. In this case, I'm just going to suggest a few improvements that I would likely consider for almost any use case.

### Choice of LLM

OpenAI's GPT models are certainly a good starting point, but there are many other LLMs available now (Llama, Falcon, Bard, Cohere, Claude, etc., just to name a few of the more well-known ones). Given the extremely fast-moving nature of the field right now, I think most serious LLM applications should include a review of current LLMs to decide which is best for the use case. Some are available through APIs, but others are available to download and host yourself, which might work better for some projects. Cost and latency are important factors to consider as well.

### Prompt Engineering

The simple prompt I created for this can most definitely be optimized. We used different techniques at my last job to refine the prompt and reduce the amount of hallucination and generally make the model's responses more relevant and of higher quality. One such technique is to use the ReAct framework with chain-of-thought prompting, which essentially prompts the LLM to form a series and thought-action pairs and reason about them to arrive at a final response. I think a next step with Docs Chat would be to integrate this type of prompting with the additional document context provided by the vector database lookup.

### RAG Enhancements

The retrieval-augmented generation here is basic and bare bones. This may be enough for many use cases, but there would certainly be challenges at scale. First of all, the document upload feature would probably become a data ingestion job instead of a web page upload. This job would likely know how to handle more than just pdf files and be customized for the types of documents most relevant to the use case. Also, my choice of using Chroma as the vector database is quite random. I would spend more time evaluating other options like Pinecone and even explore using familiar SQL- and noSQL-based databases to store and query vectors (Postgres with extensions, Redis Vector Search, etc.).

I think the document metadata that gets saved into the vector database will become more and more important for larger and more complex applications. I would expect to spend a lot more time preparing a large document dataset for ingestion. Additionally, this document similarity search technique may be augmented using other NLP techniques such as knowledge graphs that encode entities/concepts and their relationships, and that could be sent to the LLM as additional context.

### UX/UI

Chatbots have pretty well-established UX/UI patterns at this point. Docs Chat would need a professional design and branding overhaul, but I expect the main chat interaction would function very similarly (it's almost always a textbox under a scrollable message history).

## Questions? Comments?

If you'd like to discuss anything related to this project, you can reach me through email or LinkedIn.

dev@jasonbrazeal.com
https://www.linkedin.com/in/jasonbrazeal
https://jasonbrazeal.com

## Screenshots

* home page
![home page](/screenshots/home.png)

* chat before uploading documents
![chat before uploading documents](/screenshots/chat_before.png)

* chat after uploading documents
![chat after uploading documents](/screenshots/chat_after.png)

* document upload
![document upload](/screenshots/documents.png)

* chat history
![chat history](/screenshots/history.png)

* individual chat history
![individual chat history](/screenshots/history_modal.png)

* settings
![settings](/screenshots/settings.png)


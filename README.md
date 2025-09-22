# RAG Chatbot

## Todo

### Data pre-processing

1. PDF has to be converted to Markdown using `docling-serve`
2. Markdowns has to be separated based on headers, let's call it "section"
3. Detect texts and images in each section and describe images into text using VLM
4. Assign appropriate metadata to each text and image object
5. Push objects into PGVector
6. Existing data can be loaded from `arxiv-multimodal.sql` by attaching to the PSQL container and seed the database

### Data Retrieval

...

### Data Reranking

...

### Chatbot

...


### Others

List of useful commands:

1. Dump SQL data into a file

```
pg_dump -U langchain -W -h localhost langchain > arxiv-multimodal.sql
```

2. Copying the SQL data into local:

```
docker cp pgvector-rag:/arxiv-multimodal.sql ./arxiv-multimodal.sql
```
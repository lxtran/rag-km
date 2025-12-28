This Python file provides a collection of functions for the most common data ingestion methods in LlamaIndex. Each function includes a detailed docstring explaining the specific parameters required by the LlamaIndex readers.

```python
import os
from typing import List, Optional, Dict, Any, Callable
# from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext, Document # for vector index

def ingest_from_directory(
    input_dir: str, 
    recursive: bool = False, 
    required_exts: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    num_files_limit: Optional[int] = None,
    file_metadata: Optional[Callable[[str], Dict]] = None
) -> List[Document]:
    """
    Ingests documents from a local filesystem directory using SimpleDirectoryReader.
    
    Args:
        input_dir (str): Path to the directory to read from.
        recursive (bool): Whether to search subdirectories recursively. Defaults to False.
        required_exts (Optional[List[str]]): List of file extensions to include (e.g., ['.pdf', '.txt']).
        exclude (Optional[List[str]]): List of file globs to exclude from reading.
        num_files_limit (Optional[int]): Maximum number of files to read.
        file_metadata (Optional[Callable]): A function that takes a filename and returns a metadata dict.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    from llama_index.core import SimpleDirectoryReader
    
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=recursive,
        required_exts=required_exts,
        exclude=exclude,
        num_files_limit=num_files_limit,
        file_metadata=file_metadata
    )
    return reader.load_data()

def ingest_from_wikipedia(pages: List[str], lang: str = "en") -> List[Document]:
    """
    Ingests content directly from Wikipedia pages.
    
    Args:
        pages (List[str]): List of Wikipedia page titles to fetch.
        lang (str): Language code for Wikipedia. Defaults to "en".
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    from llama_index.readers.wikipedia import WikipediaReader
    
    reader = WikipediaReader()
    return reader.load_data(pages=pages, lang=lang)

def ingest_from_web_urls(urls: List[str], html_to_text: bool = True) -> List[Document]:
    """
    Ingests data from specific website URLs using SimpleWebPageReader.
    
    Args:
        urls (List[str]): A list of full URLs to scrape.
        html_to_text (bool): If True, uses 'html2text' to strip HTML tags and return clean text.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    from llama_index.readers.web import SimpleWebPageReader
    
    reader = SimpleWebPageReader(html_to_text=html_to_text)
    return reader.load_data(urls=urls)

def ingest_from_database(from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Document

    uri: str, 
    query: str, 
    metadata_cols: Optional[List[str]] = None
) -> List[Document]:
    """
    Ingests data from a SQL database by executing a query.
    
    Args:
        uri (str): Connection URI for the database (e.g., 'postgresql://user:pass@host:port/db').
        query (str): The SQL query to execute to fetch the data.
        metadata_cols (Optional[List[str]]): Columns from the query result to include as metadata.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    from llama_index.readers.database import DatabaseReader
    
    reader = DatabaseReader(uri=uri)
    return reader.load_data(query=query, metadata_cols=metadata_cols)

def ingest_from_google_drive(
    folder_id: Optional[str] = None, 
    file_ids: Optional[List[str]] = None
) -> List[Document]:
    """
    Ingests files from Google Drive. Requires 'credentials.json' for authentication.
    
    Args:
        folder_id (Optional[str]): The ID of a specific Google Drive folder to index.
        file_ids (Optional[List[str]]): A list of specific Google Drive file IDs.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    from llama_index.readers.google import GoogleDriveReader
    
    reader = GoogleDriveReader()
    return reader.load_data(folder_id=folder_id, file_ids=file_ids)

def ingest_from_slack(
    slack_token: str, 
    channel_ids: List[str], 
    earliest_date: Optional[Any] = None
) -> List[Document]:
    """
    Ingests messages from Slack channels.
    
    Args:
        slack_token (str): Slack Bot User OAuth Token.
        channel_ids (List[str]): List of Slack channel IDs to read from.
        earliest_date (Optional[datetime]): Earliest date to start reading messages from.
        
    Returns:
        List[Document]: A list of LlamaIndex Document objects.
    """
    from llama_index.readers.slack import SlackReader
    
    reader = SlackReader(slack_token=slack_token, earliest_date=earliest_date)
    return reader.load_data(channel_ids=channel_ids)

if __name__ == "__main__":
    # Example usage for directory ingestion
    # documents = ingest_from_directory("./my_docs", recursive=True)
    # print(f"Loaded {len(documents)} documents.")
    pass

```

### 📋 Setup Requirements

To run these functions, you will need to install the core library and the specific readers used in each function:

```bash
# Core library
pip install llama-index

# Specific readers
pip install llama-index-readers-wikipedia
pip install llama-index-readers-web
pip install llama-index-readers-database
pip install llama-index-readers-google
pip install llama-index-readers-slack

```

Ingesting from an existing vector database is a common pattern when you have already pre-computed embeddings (perhaps from another system) and want to use LlamaIndex's powerful query engines and agents on top of them.

In LlamaIndex, this is typically done by connecting a specific **VectorStore** (like Chroma, Pinecone, or PGVector) to a `VectorStoreIndex` using the `from_vector_store` method.

### 🐍 Vector Database Ingestion Function

This function demonstrates how to connect to an existing **ChromaDB** collection as a representative example of a vector store ingestion.

```python
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Document

def ingest_from_vector_store(
    collection_name: str,
    persist_path: str = "./chroma_db",
    embed_model: Optional[any] = None
) -> VectorStoreIndex:
    """
    Connects to an existing persistent vector store (ChromaDB) and returns an index.
    This method 'ingests' data that has already been embedded and stored.

    Args:
        collection_name (str): The name of the collection/table within the vector database.
        persist_path (str): The local directory path where the vector database is stored.
        embed_model (Optional[BaseEmbedding]): The embedding model used to create the original 
            vectors. LlamaIndex needs this to embed your new queries so they match 
            the existing vector space.

    Returns:
        VectorStoreIndex: An index object initialized from the existing vector store.
    """
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    
    # 1. Initialize the persistent vector database client
    db = chromadb.PersistentClient(path=persist_path)
    
    # 2. Get the specific collection from the database
    chroma_collection = db.get_or_create_collection(collection_name)
    
    # 3. Wrap the collection in a LlamaIndex VectorStore object
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 4. Create a storage context with this vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 5. Load the index from the vector store
    # Note: LlamaIndex doesn't "download" the docs into memory; it creates 
    # a reference to the remote/local DB for retrieval.
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    return index

# --- Example Usage ---
# from llama_index.embeddings.openai import OpenAIEmbedding
# my_index = ingest_from_vector_store(
#     collection_name="research_papers", 
#     embed_model=OpenAIEmbedding()
# )

```

### 🗝️ Why this is different from other loaders

Unlike the `SimpleDirectoryReader` or `WikipediaReader`, this method doesn't return a list of `Document` objects. Instead, it directly returns a **`VectorStoreIndex`**. This is because the data is already structured, chunked, and embedded inside the database. Returning it as raw documents would be redundant and require re-embedding them.

### 🧩 The Pipeline Concept

When using this function, the "Ingestion" step is effectively a connection establishment. Your pipeline would look like this:

1. **Connect**: Use `ingest_from_vector_store()`.
2. **Retrieve**: The index uses the `VectorStore` to perform similarity searches.
3. **Generate**: The results are passed to your LLM.

Would you like to see how to adapt this function for a cloud-based vector store like **Pinecone** or **MongoDB Atlas**?

This [LlamaIndex Vector Store tutorial](https://www.youtube.com/watch?v=i8n2Se8PAXg) explains the conceptual difference between building an index from scratch versus connecting to an existing vector storage backend.
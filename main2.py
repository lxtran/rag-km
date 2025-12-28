import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
'''

Basic RAG Code
Place data (PDFs, text files, or Word docs) into 
a folder named ./data in your project directory.
Place URL for webpages into file named ./data/urls.txt

Loading (SimpleDirectoryReader): LlamaIndex scans your folder and 
identifies file types (PDF, txt, docx). It loads the raw text into 
Document objects.

Indexing (VectorStoreIndex): This is the heart of the pipeline. 
It breaks documents into Nodes (small chunks), 
converts them into numerical Embeddings using an LLM, 
and stores them in a vector database in an in-memory database).

Querying (as_query_engine): When you ask a question, 
LlamaIndex converts your query into a vector, 
finds the most relevant "Nodes" in your index, 
and sends them along with your question to the LLM 
to generate a grounded answer.

'''

# 1. Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# 2. Load documents from the 'data' directory
documents = SimpleDirectoryReader("./data").load_data()

# 2.a In this demo, a list of URLs are stored in urls.txt. We read them in and load them as documents
#      

# 3. Create an index (automatically chunks and embeds your text)
index = VectorStoreIndex.from_documents(documents)

# 4. Create a query engine from the index
query_engine = index.as_query_engine()

# 5. Ask a question!
response = query_engine.query("What is the main topic of these documents?")
print(response)

"""
By default, the index is stored in RAM and vanishes when the script ends. 
To save it for later use: 
"""
# Save the index to a folder called 'storage'
index.storage_context.persist(persist_dir="./storage")

# To reload the index later:
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

if __name__ == "__main__":
    response = query_engine.query("What is the main topic of these documents?")
    print(response)
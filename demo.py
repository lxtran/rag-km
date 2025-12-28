import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.readers.web import SimpleWebPageReader

# 1. Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
PERSIST_DIR = "./storage"    # Directory to persist the index

def get_or_create_index():
    """Loads existing index from disk or creates a new empty one."""
    if os.path.exists(PERSIST_DIR):
        print("--- Loading existing index from storage ---")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)
    else:
        print("--- Initializing new index ---")
        # Create an empty index structure
        return VectorStoreIndex([])

def update_index_from_sources(index):
    """Checks sources for new data and updates the index."""
    all_docs = []

    # 2. Process urls.txt
    if os.path.exists("urls.txt"):
        with open("urls.txt", "r") as f:
            url_list = [line.strip() for line in f if line.strip()]
        if url_list:
            print(f"Checking {len(url_list)} URLs for updates...")
            # We assign the URL itself as the doc_id to track it uniquely
            web_docs = SimpleWebPageReader(html_to_text=True).load_data(urls=url_list)
            for doc in web_docs:
                # Use the URL as the ID so refresh knows if this specific site changed
                doc.doc_id = doc.metadata.get("url", doc.doc_id)
            all_docs.extend(web_docs)

    # 3. Process ./data folder
    if os.path.exists("./data"):
        print("Checking ./data folder for new files...")
        # filename_as_id ensures that if 'doc1.pdf' changes, it updates instead of duplicates
        local_docs = SimpleDirectoryReader("./data", filename_as_id=True).load_data()
        all_docs.extend(local_docs)

    # 4. Perform the Refresh
    # refresh_ref_docs only processes docs that are new or have changed text
    refreshed_list = index.refresh_ref_docs(all_docs)
    
    # Check if any True values exist in the list (meaning updates happened)
    if any(refreshed_list):
        print(f"Index updated! {sum(refreshed_list)} documents were added or modified.")
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("No changes detected. Index is up to date.")
    
    return index

if __name__ == "__main__":
    # Initialize the system
    current_index = get_or_create_index()
    
    # Run the update logic, ideally should be scheduled periodically
    # and run in a persistent environment separate from query handling.
    updated_index = update_index_from_sources(current_index)
    
    # Create query engine
    query_engine = updated_index.as_query_engine()
    
    # Final Question
    print("\n--- Executing Query ---")
    response = query_engine.query("What is the main topic across all these sources?")
    print(f"Response: {response}")
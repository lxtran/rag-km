from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex

# 1. Provide the specific URL(s)
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

# 2. Load documents from the URL
# html_to_text=True ensures you get clean text instead of raw HTML
documents = SimpleWebPageReader(html_to_text=True).load_data([url])

# 3. Create index and query
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What are the main risks associated with AI mentioned here?")
print(response)
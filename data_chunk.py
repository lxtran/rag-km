from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

def get_sentence_chunks(documents: list[Document], chunk_size: int = 1024, chunk_overlap: int = 20):
    """
    Splits documents into nodes based on sentence boundaries and token count.
    Standard text, books, or articles are suitable for sentence splitting.
    Args:
        documents (list): A list of LlamaIndex Document objects.
        chunk_size (int): Target number of tokens per chunk.
        chunk_overlap (int): Number of tokens to overlap between adjacent chunks.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(documents) #nodes = chunks
    return nodes

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

def get_semantic_chunks(documents: list[Document]):
    """
    Groups text into chunks based on semantic similarity using an embedding model.
    Suitable for research papers or long-form essays where topics shift significantly.

    Args:
        documents (list): A list of LlamaIndex Document objects.
        buffer_size (int): Number of sentences to group together when evaluating similarity.
        breakpoint_percentile_threshold (int): The dissimilarity threshold (0-100) to trigger a split.
    """
    embed_model = OpenAIEmbedding() # Can also use Ollama/HuggingFace
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95, 
        embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

from llama_index.core.node_parser import SentenceWindowNodeParser

def get_sentence_window_chunks(documents: list[Document]):
    """
    Splits documents into individual sentences while preserving surrounding context window.
    
    Args:
        window_size (int): Number of sentences to include before and after the target sentence.
    """
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_sentence",
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes

from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser

def get_code_chunks(documents: list[Document], language: str = "python"):
    """
    Splits raw code based on language-specific logic (e.g., functions, classes).
    
    Args:
        language (str): The programming language (e.g., 'python', 'javascript').
        chunk_lines (int): Number of lines per chunk.
    """
    splitter = CodeSplitter(
        language=language, 
        chunk_lines=40, 
        chunk_lines_overlap=15
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def get_markdown_chunks(documents: list[Document]):
    """Splits markdown based on header hierarchy (H1, H2, etc.)."""
    parser = MarkdownNodeParser()
    return parser.get_nodes_from_documents(documents)

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

def get_wiki_parent_child_chunks(documents):
    """
    Creates a hierarchy: Large Chunks (Section) -> Small Chunks (Paragraphs).
    Search happens on the Paragraph level, but the Section is sent to the LLM.
    """
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128] # Three levels of granularity
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # We only index the smallest 'leaf' nodes (128 tokens)
    leaf_nodes = get_leaf_nodes(nodes)
    return leaf_nodes

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

def get_wiki_parent_child_chunks(documents):
    """
    Creates a hierarchy: Large Chunks (Section) -> Small Chunks (Paragraphs).
    Search happens on the Paragraph level, but the Section is sent to the LLM.
    """
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128] # Three levels of granularity
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # We only index the smallest 'leaf' nodes (128 tokens)
    leaf_nodes = get_leaf_nodes(nodes)
    return leaf_nodes

if __name__ == "__main__":
    pass

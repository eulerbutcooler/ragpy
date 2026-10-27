
from typing import cast

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


def chunk_documents(documents: list[Document]) -> list[TextNode]:
    splitter = SentenceSplitter(
        chunk_size=768,
        chunk_overlap=40,
        include_metadata=True,
        include_prev_next_rel=True,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
    return cast(list[TextNode], nodes)


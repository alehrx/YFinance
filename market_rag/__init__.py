"""Market RAG: retrieval-augmented mapping of (news + technical/valuation metrics)
to forward gain/loss for an index or ticker.

Each trading day becomes a "document" in a knowledge base: a snapshot of metrics
(price vs 50/200-day averages, 1-year return, P/E, CAPE, VIX, ...) plus a digest
of the headlines from the preceding week, labelled with what the index did
over the next N trading days. A query day is answered by retrieving the most
similar *past* days (whose outcome was already known on the query date) and
aggregating their outcomes; optionally an LLM writes the final call from the
retrieved analogs.
"""

from .features import FEATURE_COLUMNS, build_features
from .rag import MarketRAG, RAGConfig

__all__ = ["FEATURE_COLUMNS", "build_features", "MarketRAG", "RAGConfig"]

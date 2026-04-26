"""
CloudSRE v2 — Search Service (port 8007).

Full-text search indexing service (Elasticsearch-like).
Cascade pattern: DB lock → search index can't refresh → stale results → user complaints.

Fault types:
  - index_corruption: Search index becomes corrupt, returns wrong results
  - index_lag: Index falls behind by 1000+ documents
"""

import time
from fastapi import Request
from fastapi.responses import JSONResponse
from services.base_service import BaseService


class SearchService(BaseService):
    def __init__(self, port: int = 8007, log_dir: str = "/var/log"):
        super().__init__("search", port=port, log_dir=log_dir)
        self._index = {}
        self._indexed_count = 0
        self._query_count = 0
        self._lag = 0
        self._corrupted = False
        self._seed_index()
        self._register_routes()

    def _seed_index(self):
        now = time.time()
        docs = {
            "doc:product_1": {"title": "Premium Widget", "body": "High-quality widget", "indexed_at": now},
            "doc:product_2": {"title": "Basic Widget", "body": "Standard widget", "indexed_at": now},
            "doc:order_1001": {"title": "Order #1001", "body": "Shipped to user_1", "indexed_at": now},
            "doc:user_profile_1": {"title": "User Profile", "body": "John Doe account", "indexed_at": now},
            "doc:faq_billing": {"title": "Billing FAQ", "body": "How to update payment", "indexed_at": now},
        }
        self._index = docs
        self._indexed_count = len(docs)
        self._query_count = 312

    def _register_routes(self):
        @self.app.get("/search/query")
        async def search_query(q: str = ""):
            self._query_count += 1
            if self._corrupted:
                self.logger.error(f"Search returned corrupted results for: {q}")
                return {"query": q, "results": [], "error": "INDEX_CORRUPTED", "total": 0}
            results = [v for k, v in self._index.items() if q.lower() in v.get("body", "").lower()]
            return {"query": q, "results": results[:10], "total": len(results)}

        @self.app.get("/search/stats")
        async def search_stats():
            return {
                "service": "search", "indexed_docs": self._indexed_count,
                "query_count": self._query_count, "index_lag": self._lag,
                "corrupted": self._corrupted, "index_size": len(self._index),
            }

        @self.app.post("/search/reindex")
        async def reindex():
            self._seed_index()
            self._corrupted = False
            self._lag = 0
            self.logger.info("Search index rebuilt successfully")
            return {"status": "reindexed", "docs": self._indexed_count}

    def inject_corruption(self):
        self._corrupted = True
        self.set_degraded()
        self.logger.error("FAULT: Search index corrupted — returning empty results")

    def inject_lag(self):
        self._lag = 1247
        self.set_degraded()
        self.logger.error("FAULT: Search index lagging by 1247 documents")

    def reset(self):
        super().reset()
        self._index.clear()
        self._corrupted = False
        self._lag = 0
        self._query_count = 0
        self._seed_index()

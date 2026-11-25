"""Read-only MCP retrieval target. Indexes documents, never evaluator labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .retrieval import ScopedRetriever


class Hit(BaseModel):
    id: str
    score: float


class SearchResponse(BaseModel):
    results: list[Hit]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--documents", type=Path, required=True)
    p.add_argument("--method", choices=("bm25", "tfidf"), required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--index-report", type=Path, required=True)
    args = p.parse_args()
    retriever = ScopedRetriever(args.documents, method=args.method)
    args.index_report.parent.mkdir(parents=True, exist_ok=True)
    args.index_report.write_text(json.dumps({
        "method": args.method, "build_stats": retriever.build_stats,
        "execution": "single process, synchronous retrieval in the MCP event loop",
        "response": "document IDs and scores only; no context text",
        "query_result_cache": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    server = FastMCP("Public memory retrieval baseline", host="127.0.0.1",
                     port=args.port, stateless_http=True, json_response=True,
                     log_level="ERROR")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
                 structured_output=True)
    async def memory_search(
        query: str,
        top_k: Annotated[int, Field(ge=1, le=1000)] = 20,
        scope: str = "",
    ) -> SearchResponse:
        """Retrieve ranked evidence IDs from the requested isolated corpus."""
        return SearchResponse(results=[Hit(**hit) for hit in
                                       retriever.search(query, top_k, scope)])

    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()

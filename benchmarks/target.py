"""Loopback, read-only MCP adapter for fixed duorou retrieval experiments.

The configured principal is a local benchmark service identity. This endpoint
has no remote-user authentication and must not be exposed as a production API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .retriever import DuorouRetriever, METHODS


class Hit(BaseModel):
    id: str
    score: float


class SearchResponse(BaseModel):
    results: list[Hit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--index-report", type=Path, required=True)
    args = parser.parse_args()
    retriever = DuorouRetriever(args.db, args.method, args.documents)
    args.index_report.parent.mkdir(parents=True, exist_ok=True)
    args.index_report.write_text(json.dumps({
        "method": args.method, "build_stats": retriever.build_stats,
        "execution": "single process, synchronous retrieval in MCP event loop",
        "response": "document IDs and scores only; no context text",
        "query_result_cache": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    server = FastMCP("duorou retrieval benchmark target", host="127.0.0.1",
                     port=args.port, stateless_http=True, json_response=True,
                     log_level="ERROR")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
                 structured_output=True)
    async def memory_search(
        query: str,
        top_k: Annotated[int, Field(ge=1, le=100)] = 20,
        scope: str = "",
    ) -> SearchResponse:
        """Retrieve ranked evidence IDs from an explicitly authorized scope."""
        return SearchResponse(results=[Hit(**hit) for hit in
                                       retriever.search(query, top_k, scope)])

    try:
        server.run(transport="streamable-http")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()

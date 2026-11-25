"""MCP is a control plane; the load loop never invokes an LLM."""
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .jobs import JobManager
from .models import RunConfig


def create_server(manager: JobManager) -> FastMCP:
    @asynccontextmanager
    async def lifespan(server):
        try:
            yield manager
        finally:
            await manager.close()

    mcp = FastMCP("memory-query-bench", lifespan=lifespan,
                  instructions="Independent query benchmark. Select configured targets. Start returns a job ID; poll status and fetch report. Synthetic results only validate the harness. Never interpret success-only latency without failure/quality counters.")
    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    load = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)

    @mcp.tool(annotations=read)
    def memory_bench_targets() -> dict[str, Any]:
        """List operator-configured targets, datasets and limits; never exposes tokens or commands."""
        return manager.registry.describe()

    @mcp.tool(annotations=load)
    async def memory_bench_start(config: RunConfig) -> dict[str, Any]:
        """Start bounded query load. Sends real queries, consumes target resources, and saves local reports. No target data is explicitly written by this tool; the target query may have its own side effects."""
        return manager.start(config)

    @mcp.tool(annotations=read)
    def memory_bench_status(run_id: str) -> dict[str, Any]:
        """Read job state and progress; poll sparingly, outside the load loop."""
        return manager.status(run_id)

    @mcp.tool(annotations=read)
    def memory_bench_report(run_id: str) -> dict[str, Any]:
        """Get aggregate metrics and provenance. Detailed query text is not returned or logged."""
        return manager.report(run_id)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    async def memory_bench_cancel(run_id: str) -> dict[str, Any]:
        """Request cooperative cancellation. Already dispatched operations may still execute on the target."""
        return manager.cancel(run_id)

    @mcp.tool(annotations=read)
    def memory_bench_compare(run_ids: list[str]) -> dict[str, Any]:
        """Compare completed runs only when replay workload fingerprints match."""
        return manager.compare(run_ids)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True))
    async def memory_bench_visibility_start(target_id: str, scope: str = "bench-visibility",
                                            timeout_ms: int = 5000, poll_interval_ms: int = 50) -> dict[str, Any]:
        """Opt-in write/read/delete sentinel test in a bench-* namespace. Requires operator-enabled writes. Deletes only its random sentinel; reports cleanup failures."""
        return manager.start_visibility(target_id, scope, timeout_ms, poll_interval_ms)

    return mcp

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import Registry
from .jobs import JobManager
from .models import RunConfig


async def run_cli(args):
    manager = JobManager(Registry(args.config), args.output)
    try:
        cfg = RunConfig.model_validate_json(Path(args.run_config).read_text()) if args.run_config else RunConfig(
            target_id=args.target, requests=args.requests, concurrency=args.concurrency,
            mode=args.mode, rate_per_second=args.rate, warmup=args.warmup,
            timeout_ms=args.timeout_ms, queue_capacity=args.queue_capacity)
        started = manager.start(cfg)
        await manager.jobs[started["run_id"]].task
        report = manager.report(started["run_id"])
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("status") == "completed" else 2
    finally:
        await manager.close()


def main():
    parser = argparse.ArgumentParser(description="Independent memory-query MCP benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", help="operator target/dataset registry JSON")
        p.add_argument("--output", default="reports")
        if name == "run":
            p.add_argument("--run-config", help="full RunConfig JSON file")
            p.add_argument("--target", default="mock")
            p.add_argument("--requests", type=int, default=500)
            p.add_argument("--concurrency", type=int, default=4)
            p.add_argument("--mode", choices=["closed_loop", "open_loop"], default="closed_loop")
            p.add_argument("--rate", type=float, default=50)
            p.add_argument("--warmup", type=int, default=20)
            p.add_argument("--timeout-ms", type=int, default=2000)
            p.add_argument("--queue-capacity", type=int, default=64)
    args = parser.parse_args()
    if args.command == "serve":
        from .server import create_server
        create_server(JobManager(Registry(args.config), args.output)).run(transport="stdio")
    else:
        raise SystemExit(asyncio.run(run_cli(args)))


if __name__ == "__main__":
    main()

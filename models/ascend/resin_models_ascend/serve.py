"""Serve a local Qwen checkpoint on one or more Ascend NPUs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist

from resin import Engine
from resin.backend.ascend.graphs import GRAPH_MODES
from resin.backend.ascend.runtime import create_groups, initialize_device
from resin.backend.distributed import DistributedEngine, DistributedWorker
from resin.recipe import get_native_recipe
from resin.server import HttpServer
from resin_models_ascend.model import TensorParallel
from resin_models_ascend.runner import QwenAscendRunner

log = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--recipe", choices=("qwen3_5", "qwen3_8"), default="qwen3_5")
    parser.add_argument("--model-name", default="qwen-ascend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--cache-pages", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=256)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--quantization", choices=("bf16", "w8a8"), default="bf16")
    parser.add_argument("--attention", choices=("torch", "native"), default="torch")
    parser.add_argument("--native-gdn-prefill", action="store_true")
    parser.add_argument(
        "--fused-rmsnorm",
        action="store_true",
        help="use the TorchNPU RMSNorm kernel",
    )
    parser.add_argument(
        "--triton-gdn",
        action="store_true",
        help="use fused Triton Ascend recurrence for decode and MTP verify",
    )
    parser.add_argument("--graph", choices=GRAPH_MODES, default="eager")
    parser.add_argument("--graph-max-pages", type=int, default=32)
    parser.add_argument(
        "--graph-capture-pages",
        type=int,
        nargs="+",
        help="pre-capture these KV page buckets before serving with --graph aclgraph",
    )
    parser.add_argument(
        "--graph-verify-batches",
        type=int,
        nargs="+",
        help="also pre-capture MTP verify graphs for these exact request counts",
    )
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        help="enable MTP and draft this many tokens by reusing the same MTP head",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.num_speculative_tokens is not None:
        if not 1 <= args.num_speculative_tokens <= args.context_length:
            parser.error(
                "--num-speculative-tokens must be positive and at most --context-length"
            )
        args.mtp = True
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if (
        min(args.tp, args.dp) < 1
        or args.tp * args.dp != world
        or local_world != world
        or world > 8
    ):
        parser.error("launch a single node with DP × TP processes, up to eight ranks")
    if args.graph != "eager" and (world != 1 or args.attention != "torch"):
        parser.error(f"{args.graph} requires one card and --attention torch")
    if args.mtp and args.graph == "npugraph_ex":
        parser.error("MTP graph execution requires --graph aclgraph")
    if args.graph_capture_pages is not None and args.graph != "aclgraph":
        parser.error("--graph-capture-pages requires --graph aclgraph")
    if args.graph_verify_batches is not None and (
        not args.mtp or args.graph_capture_pages is None
    ):
        parser.error("--graph-verify-batches requires MTP and --graph-capture-pages")
    if (
        min(
            args.context_length,
            args.cache_pages,
            args.prefill_chunk,
            args.max_requests,
            args.graph_max_pages,
        )
        < 1
    ):
        parser.error(
            "context, cache, prefill, request, and graph capacities must be positive"
        )
    checkpoint = args.checkpoint.expanduser().resolve()
    device = initialize_device(rank)
    groups = create_groups(args.dp, args.tp, world) if world > 1 else None
    try:
        tp = (
            TensorParallel()
            if groups is None
            else TensorParallel(
                groups.attn_tp.size,
                groups.attn_tp.local_rank,
                groups.attn_tp.handle,
            )
        )
        try:
            runner = QwenAscendRunner(
                checkpoint,
                device=device,
                tp=tp,
                cache_pages=args.cache_pages,
                prefill_chunk=args.prefill_chunk,
                mtp=args.mtp,
                num_speculative_tokens=args.num_speculative_tokens,
                quantization=args.quantization,
                graph=args.graph,
                graph_max_pages=args.graph_max_pages,
                graph_capture_pages=args.graph_capture_pages,
                graph_verify_batches=args.graph_verify_batches,
                attention=args.attention,
                native_gdn_prefill=args.native_gdn_prefill,
                fused_rmsnorm=args.fused_rmsnorm,
                triton_gdn=args.triton_gdn,
            )
            if args.context_length > runner.model.config.max_position_embeddings:
                raise ValueError("context length exceeds checkpoint position capacity")
            if (
                runner.layout.estimated_parent_count(
                    [args.context_length + max(0, runner.num_speculative_tokens - 1)]
                )
                > args.cache_pages
            ):
                raise ValueError(
                    "cache pages cannot hold one request and its speculative lookahead "
                    "at the configured context length"
                )
            recipe = None
            if groups is None or groups.is_driver:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, local_files_only=True
                )
                recipe = get_native_recipe(args.recipe, tokenizer, text_only=True)
            failure = None
        except Exception as exc:
            if groups is None:
                raise
            failure = f"rank {rank}: {exc}"
        if groups is not None:
            failures = [None] * world
            dist.all_gather_object(failures, failure)
            if any(failures):
                raise RuntimeError(
                    f"Ascend startup failed: {[item for item in failures if item]}"
                )

        options = dict(
            context_length=args.context_length,
            max_running_requests=args.max_requests,
            enable_overlap=False,
            prefix_cache=False,
        )
        if groups is None:
            with Engine(runner, **options) as engine:
                HttpServer(engine, recipe=recipe, model=args.model_name).serve(
                    host=args.host,
                    port=args.port,
                )
        else:
            worker = DistributedWorker(runner, groups)
            if groups.is_driver:
                engine = DistributedEngine(worker, **options)
                try:
                    HttpServer(engine, recipe=recipe, model=args.model_name).serve(
                        host=args.host,
                        port=args.port,
                    )
                finally:
                    engine.close()
            else:
                worker.serve()
        log.info(
            "rank=%d peak_allocated_bytes=%d graphs=%d replays=%d fallbacks=%d drafts=%d accepted=%d",
            rank,
            torch.npu.max_memory_allocated(device),
            len(runner.graphs),
            runner.graph_replays,
            runner.graph_fallbacks,
            runner.drafted,
            runner.accepted,
        )
    finally:
        if groups is not None:
            try:
                groups.close()
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    main()

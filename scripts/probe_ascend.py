"""Check a preconfigured Ascend host using only its installed Python packages."""

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path


class InterfaceMissing(Exception):
    pass


class NumericalMismatch(Exception):
    pass


def require(module, name):
    value = getattr(module, name, None)
    if value is None:
        raise InterfaceMissing(name)
    return value


def check(torch, actual, expected, *, atol=0.0, rtol=0.0):
    actual, expected = actual.cpu().float(), expected.cpu().float()
    if actual.shape != expected.shape or not torch.allclose(
        actual, expected, atol=atol, rtol=rtol
    ):
        raise NumericalMismatch(
            f"actual shape {tuple(actual.shape)}, expected {tuple(expected.shape)}; "
            f"first values {actual.flatten()[:8].tolist()} vs "
            f"{expected.flatten()[:8].tolist()}"
        )


def run_command(args, timeout):
    started = time.monotonic()
    if shutil.which(args[0]) is None:
        return {"status": "interface_missing", "error": args[0]}
    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            status = "pass" if process.returncode == 0 else "execution_failed"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            status = "timeout"
    return {
        "status": status,
        "returncode": process.returncode,
        "seconds": round(time.monotonic() - started, 3),
        "stdout": stdout.decode(errors="replace")[-12000:],
        "stderr": stderr.decode(errors="replace")[-12000:],
    }


def inventory(timeout):
    variables = {
        name: os.environ.get(name)
        for name in (
            "ASCEND_HOME_PATH",
            "ASCEND_OPP_PATH",
            "ASCEND_CUSTOM_OPP_PATH",
            "ASCEND_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
        )
    }
    roots = [Path("/usr/local/Ascend")]
    roots += [
        Path(variables[name]).resolve()
        for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH")
        if variables[name]
    ]
    patterns = (
        "version.info",
        "*_install.info",
        "*/version.info",
        "*/*/version.info",
        "*/version.cfg",
        "ascend_toolkit_install.info",
        "opp*/**/version.info",
    )
    paths = {Path("/etc/ascend_install.info")}
    for root in roots:
        for pattern in patterns:
            paths.update(sorted(root.glob(pattern))[:32])
    files = {}
    for path in sorted(paths):
        try:
            if not path.is_file() or path.stat().st_size > 32768:
                continue
            # Install info files may also include usernames and paths.
            lines = [
                line
                for line in path.read_text(errors="replace").splitlines()
                if "version" in line.partition("=")[0].partition(":")[0].lower()
            ]
        except OSError as exc:
            files[str(path)] = [str(exc)]
        else:
            if lines:
                files[str(path)] = lines
    packages = {}
    for name in ("torch", "torch-npu", "torchair", "triton-ascend", "fla-npu", "resin"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commands = {
        "npu_smi": ["npu-smi", "info"],
        "npu_mapping": ["npu-smi", "info", "-m"],
        "asys_hardware": ["asys", "info", "-r=hardware"],
    }
    query = next(
        (
            root / "query_pkg_version.sh"
            for root in roots
            if (root / "query_pkg_version.sh").is_file()
        ),
        None,
    )
    if query is not None:
        commands["cann_packages"] = ["bash", str(query)]
    # Keep only SoC/architecture evidence from asys; its full output may
    # include host inventory such as serial numbers.
    results = {
        name: run_command(cmd, min(timeout, 15)) for name, cmd in commands.items()
    }
    results["asys_hardware"]["stdout"] = "\n".join(
        line
        for line in results["asys_hardware"].get("stdout", "").splitlines()
        if any(
            key in line.lower() for key in ("soc", "chip info", "arch info", "npu arch")
        )
    )
    return {
        "platform": {
            "os": platform.system(),
            "arch": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": packages,
        "environment": variables,
        "version_files": files,
        "commands": results,
        "tools": {
            name: shutil.which(name)
            for name in (
                "c++",
                "cmake",
                "ninja",
                "bisheng",
                "msprof",
                "rustc",
                "cargo",
                "maturin",
            )
        },
    }


def library_report():
    libraries = {}
    for library in (
        "libascendcl.so",
        "libopapi.so",
        "libnnopbase.so",
        "libhccl.so",
        "libatb.so",
    ):
        try:
            ctypes.CDLL(library)
        except OSError as exc:
            libraries[library] = str(exc)
        else:
            libraries[library] = "loaded"
    required = ("libascendcl.so", "libopapi.so", "libnnopbase.so")
    status = (
        "pass" if all(libraries[key] == "loaded" for key in required) else "load_failed"
    )
    return {"status": status, "details": libraries}


def smoke(torch, npu, name, device):
    torch.npu.set_device(device)
    dtype = torch.bfloat16
    if name == "device":
        devices = []
        for index in range(torch.npu.device_count()):
            props = torch.npu.get_device_properties(index)
            devices.append(
                {"index": index, "name": props.name, "memory_bytes": props.total_memory}
            )
        return {
            "devices": devices,
            "torch": torch.__version__,
            "torch_npu": npu.__version__,
        }
    if name == "bf16_stream":
        stream, event = torch.npu.Stream(), torch.npu.Event()
        with torch.npu.stream(stream):
            matrix = torch.ones((16, 16), device=device, dtype=dtype)
            output = matrix @ matrix
            event.record(stream)
        torch.npu.current_stream().wait_event(event)
        check(torch, output, torch.full((16, 16), 16))
    elif name == "kv_write":
        keys = torch.zeros((2, 128, 1, 256), dtype=dtype, device=device)
        values = torch.zeros_like(keys)
        row = torch.ones((1, 1, 256), dtype=dtype, device=device)
        slot = torch.tensor([129], dtype=torch.int32, device=device)
        require(npu, "npu_scatter_pa_kv_cache")(
            row, 2 * row, keys, values, slot, cache_mode="Norm"
        )
        check(torch, keys[1, 1], row[0])
        check(torch, values[1, 1], 2 * row[0])
        check(torch, values[0, 1], torch.zeros_like(row[0]))
    elif name == "paged_attention":
        q = torch.ones((1, 4, 256), dtype=dtype, device=device)
        k = torch.zeros((2, 128, 256), dtype=dtype, device=device)
        v = torch.zeros_like(k)
        v[1, 0] = 3
        table = torch.tensor([[1]], dtype=torch.int32, device=device)
        output, _ = require(npu, "npu_fused_infer_attention_score")(
            q,
            k,
            v,
            input_layout="TND",
            num_heads=4,
            num_key_value_heads=1,
            scale=256**-0.5,
            block_table=table,
            block_size=128,
            actual_seq_lengths=[1],
            actual_seq_lengths_kv=[1],
            sparse_mode=0,
        )
        check(torch, output, torch.full_like(q, 3))
    elif name in ("gdn_recurrent", "gdn_chunk"):
        tokens = 1 if name == "gdn_recurrent" else 64
        q = torch.zeros((tokens, 1, 128), dtype=dtype, device=device)
        q[:, :, 0] = 1
        state = torch.zeros((1, 1, 128, 128), dtype=dtype, device=device)
        beta = torch.ones((tokens, 1), dtype=dtype, device=device)
        lengths = torch.tensor([tokens], dtype=torch.int32, device=device)
        decay = torch.zeros((tokens, 1), dtype=torch.float32, device=device)
        kwargs = dict(beta=beta, scale=128**-0.5, actual_seq_lengths=lengths, g=decay)
        if tokens == 1:
            output = require(npu, "npu_recurrent_gated_delta_rule")(
                q,
                q,
                q,
                state,
                ssm_state_indices=torch.zeros(1, dtype=torch.int32, device=device),
                **kwargs,
            )
        else:
            output, state = require(npu, "npu_chunk_gated_delta_rule")(
                q, q, q, initial_state=state, **kwargs
            )
        check(torch, output, q * (128**-0.5), atol=0.002)
        check(torch, state[0, 0, :1, :], q[:1, 0])
    elif name == "w8a8":
        x = torch.ones((16, 128), device=device, dtype=dtype)
        quant, scales = require(npu, "npu_dynamic_quant")(x, dst_type=torch.int8)
        weight = torch.ones((128, 64), device=device, dtype=torch.int8)
        scale = torch.ones(64, device=device, dtype=torch.float32)
        output = require(npu, "npu_quant_matmul")(
            quant, weight, scale, pertoken_scale=scales, output_dtype=dtype
        )
        check(torch, output, torch.full((16, 64), 128), atol=1)
    elif name in ("aot_eager", "npugraph_ex", "aclgraph"):
        x = torch.ones((16, 16), device=device, dtype=dtype)
        if name == "aclgraph":
            graph = require(torch.npu, "NPUGraph")()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                _ = x * 2 + 1
            torch.npu.current_stream().wait_stream(stream)
            with require(torch.npu, "graph")(graph, stream=stream):
                output = x * 2 + 1
            for value in (1, 3, 5):
                x.fill_(value)
                graph.replay()
                check(torch, output, torch.full((16, 16), value * 2 + 1))
        else:
            if name == "npugraph_ex":
                try:
                    importlib.import_module("torchair")
                except ModuleNotFoundError as exc:
                    if exc.name != "torchair":
                        raise
                if name not in torch._dynamo.list_backends(exclude_tags=()):
                    # Some releases lazily register this backend on first lookup.
                    try:
                        torch._dynamo.lookup_backend(name)
                    except torch._dynamo.exc.InvalidBackend as exc:
                        raise InterfaceMissing(name) from exc
            compiled = torch.compile(
                lambda arg: arg * 2 + 1, backend=name, fullgraph=True, dynamic=False
            )
            for value in (1, 3, 5):
                x.fill_(value)
                check(torch, compiled(x), torch.full((16, 16), value * 2 + 1))
    torch.npu.synchronize(device)
    return {}


def hccl_rank(rank, ranks, path, first_device):
    import torch
    import torch.distributed as dist
    import torch_npu

    del torch_npu
    device = f"npu:{first_device + rank}"
    torch.npu.set_device(device)
    timeout = timedelta(seconds=45)
    dist.init_process_group(
        "gloo",
        init_method=Path(path).as_uri(),
        rank=rank,
        world_size=ranks,
        timeout=timeout,
    )
    try:
        group = dist.new_group(list(range(ranks)), backend="hccl", timeout=timeout)
        try:
            tensor = torch.full((16,), rank + 1, dtype=torch.float32, device=device)
            dist.all_reduce(tensor, group=group)
            check(torch, tensor, torch.full((16,), ranks * (ranks + 1) // 2))
            dist.barrier()
        finally:
            dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


def worker(name, index, ranks):
    if name == "libraries":
        print("RESIN_PROBE_RESULT=" + json.dumps(library_report()), flush=True)
        return
    try:
        try:
            import torch
            import torch_npu
        except (ImportError, OSError) as exc:
            result = {"status": "load_failed", "error": str(exc)}
        else:
            with torch.inference_mode():
                if name == "hccl":
                    with tempfile.TemporaryDirectory(prefix="resin-hccl-") as directory:
                        torch.multiprocessing.spawn(
                            hccl_rank,
                            args=(ranks, str(Path(directory) / "store"), index),
                            nprocs=ranks,
                        )
                    details = {"ranks": ranks}
                else:
                    details = smoke(torch, torch_npu, name, f"npu:{index}")
            result = {"status": "pass", "details": details}
    except InterfaceMissing as exc:
        result = {"status": "interface_missing", "error": str(exc)}
    except NumericalMismatch as exc:
        result = {"status": "numerical_mismatch", "error": str(exc)}
    except (ImportError, OSError) as exc:
        result = {"status": "load_failed", "error": str(exc)}
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    print("RESIN_PROBE_RESULT=" + json.dumps(result), flush=True)


CHECKS = (
    "libraries",
    "device",
    "bf16_stream",
    "kv_write",
    "paged_attention",
    "gdn_recurrent",
    "gdn_chunk",
    "w8a8",
    "aot_eager",
    "npugraph_ex",
    "aclgraph",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ascend-probe.json"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hccl-ranks", type=int, choices=(0, 2, 4, 8), default=0)
    parser.add_argument(
        "--timeout", type=int, default=120, help="seconds per runtime check"
    )
    parser.add_argument("--worker", choices=(*CHECKS, "hccl"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.timeout < 1:
        parser.error("device must be nonnegative and timeout must be positive")
    if args.worker:
        worker(args.worker, args.device, args.hccl_ranks)
        return 0
    report = {"schema_version": 1, "inventory": inventory(args.timeout), "checks": {}}
    names = (*CHECKS, "hccl") if args.hccl_ranks else CHECKS
    for name in names:
        result = run_command(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                name,
                "--device",
                str(args.device),
                "--hccl-ranks",
                str(args.hccl_ranks),
            ],
            args.timeout,
        )
        logs = result.get("stdout", "").splitlines()
        marker = "RESIN_PROBE_RESULT="
        if result["status"] == "pass":
            payload = next(
                (
                    line[len(marker) :]
                    for line in reversed(logs)
                    if line.startswith(marker)
                ),
                None,
            )
            if payload is None:
                result.update(
                    status="execution_failed", error="worker produced no result"
                )
            else:
                result.update(json.loads(payload))
                result["stdout"] = "\n".join(
                    line for line in logs if not line.startswith(marker)
                )
        report["checks"][name] = result
        print(f"{name}: {result['status']}", file=sys.stderr, flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")
    required = ("device", "bf16_stream", *(("hccl",) if args.hccl_ranks else ()))
    return int(any(report["checks"][name]["status"] != "pass" for name in required))


if __name__ == "__main__":
    sys.exit(main())

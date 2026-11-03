"""CPU regressions for NPU patches applied to supplied upstream source trees.

Set VLLM_SOURCE_DIR and VLLM_ASCEND_SOURCE_DIR to clean source repositories.
Only copies of the files at HEAD are patched; neither source tree is modified.
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
NUM_GPUS = 0


def _load_definitions(path, names, namespace, class_name=None, base=object):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing definitions in {path}: {names}")
    if class_name:
        nodes = [
            ast.ClassDef(
                name="Subject", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=nodes, decorator_list=[]
            )
        ]
        namespace["Base"] = base
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace.get("Subject")


class TestNPUWeightPatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sources = {"vllm": os.getenv("VLLM_SOURCE_DIR"), "vllm-ascend": os.getenv("VLLM_ASCEND_SOURCE_DIR")}
        if not all(sources.values()):
            raise unittest.SkipTest("Set VLLM_SOURCE_DIR and VLLM_ASCEND_SOURCE_DIR to run source-patch regressions")
        cls.temporary = tempfile.TemporaryDirectory(prefix="vime-patches-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.sources = {}
        for name, source in sources.items():
            target = Path(cls.temporary.name) / name
            target.mkdir()
            patch_path = REPO / "docker/npu_patch" / f"{name}.patch"
            for relative in dict.fromkeys(
                re.findall(r"^--- a/(.+)$", patch_path.read_text(encoding="utf-8"), re.MULTILINE)
            ):
                data = subprocess.check_output(["git", "-C", source, "show", f"HEAD:{relative}"])
                file = target / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)
            subprocess.run(["git", "init", "-q", str(target)], check=True, capture_output=True)
            for arguments in (["--check", "--whitespace=error"], ["--whitespace=error"]):
                subprocess.run(
                    ["git", "-C", str(target), "apply", *arguments, str(patch_path)], check=True, capture_output=True
                )
            for file in target.rglob("*.py"):
                ast.parse(file.read_text(encoding="utf-8"))
            cls.sources[name] = target

    def _worker(self):
        namespace = {"set_current_vllm_config": lambda config: contextlib.nullcontext()}
        subject = _load_definitions(
            self.sources["vllm-ascend"] / "vllm_ascend/worker/worker.py",
            {
                "start_weight_update",
                "_start_weight_update",
                "update_weights",
                "finish_weight_update",
                "start_draft_weight_update",
                "get_draft_model",
                "_set_draft_weight_update_target",
            },
            namespace,
            "NPUWorker",
        )
        worker = subject()
        worker._check_weight_transfer_engine = lambda: None
        worker._check_nz_disabled = lambda: None
        worker._weight_update_active = False
        worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=1, world_size=2))
        worker.rank = 1
        worker.calls = []
        worker.model_runner = SimpleNamespace(
            reset_lora_state=lambda: worker.calls.append("reset_lora"), get_draft_model=lambda: "draft"
        )
        worker.speculative_config = SimpleNamespace(draft_model_config="draft_config")
        worker.weight_transfer_engine = SimpleNamespace(
            supports_draft_weight_update=True,
            start_weight_update=lambda: worker.calls.append("start"),
            update_weights=lambda payload: worker.calls.append(payload),
            finish_weight_update=lambda: worker.calls.append("finish"),
            set_weight_update_target=lambda *args: worker.calls.append(args),
            reset_weight_update_target=lambda: worker.calls.append("reset_target"),
        )
        return worker

    def test_rank_local_payload_uses_global_worker_index(self):
        worker = self._worker()
        worker.start_weight_update()
        payloads = [{"rank": index} for index in range(4)]
        worker.update_weights(payloads)
        worker.finish_weight_update()
        self.assertEqual(worker.calls, ["start", {"rank": 3}, "finish", "reset_lora", "reset_target"])
        self.assertFalse(worker._weight_update_active)

    def test_draft_target_is_reset_when_finish_fails(self):
        worker = self._worker()
        worker.start_draft_weight_update()
        self.assertEqual(worker.calls[:2], [("draft", "draft_config"), "start"])

        def fail():
            raise RuntimeError("reload failed")

        worker.weight_transfer_engine.finish_weight_update = fail
        with self.assertRaisesRegex(RuntimeError, "reload failed"):
            worker.finish_weight_update()
        self.assertFalse(worker._weight_update_active)
        self.assertEqual(worker.calls[-1], "reset_target")

    def test_empty_rank_payload_does_not_open_an_ipc_handle(self):
        subject = _load_definitions(
            self.sources["vllm-ascend"] / "vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py",
            {"receive_weights"},
            {},
            "NPUIPCWeightTransferEngine",
        )
        # No device, model or torch_npu is supplied: empty rank payloads must
        # return before resolving a physical device or opening shared storage.
        subject().receive_weights(SimpleNamespace(names=[]))

    def test_ipc_collection_happens_after_the_parent_releases_references(self):
        calls = []

        class Parent:
            def send_weights(self):
                calls.append("send_and_release")

        namespace = {
            "torch": SimpleNamespace(
                npu=SimpleNamespace(
                    ipc_collect=lambda: calls.append("collect"), empty_cache=lambda: calls.append("empty_cache")
                )
            )
        }
        subject = _load_definitions(
            self.sources["vllm-ascend"] / "vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py",
            {"send_weights"},
            namespace,
            "NPUIPCTrainerWeightTransferEngine",
            Parent,
        )
        subject().send_weights()
        self.assertEqual(calls, ["send_and_release", "collect", "empty_cache"])

    def test_legacy_response_fields_and_native_metrics_coexist(self):
        def record(**kwargs):
            return SimpleNamespace(**kwargs)

        stats = {"num_draft_tokens": 4, "num_accepted_draft_tokens": 3, "num_spec_steps": 1}
        build_metrics = Mock(return_value=SimpleNamespace(model_dump=lambda: stats))
        namespace = {
            "UsageInfo": record,
            "PerRequestMetrics": record,
            "build_spec_decoding_metrics": build_metrics,
            "clamp_prompt_logprobs": lambda value: value,
        }
        subject = _load_definitions(
            self.sources["vllm"] / "vllm/entrypoints/scale_out/token_in_token_out/serving.py",
            {"_build_response_fields"},
            namespace,
            "ServingTokens",
        )
        server = subject()
        server.enable_prompt_tokens_details = False
        request = SimpleNamespace(
            sampling_params=SimpleNamespace(n=1), return_token_ids=False, _response_mm_placeholders=None
        )
        result = SimpleNamespace(
            prompt_token_ids=[1, 2],
            encoder_prompt_token_ids=None,
            prompt_logprobs=None,
            prompt_token_id_logprobs=None,
            kv_transfer_params=None,
            ec_transfer_params=None,
        )
        fields = server._build_response_fields(request, result, "request", "model", 0, 1, SimpleNamespace(), "7")
        self.assertEqual(fields["weight_version"], "7")
        self.assertEqual(fields["request_spec_decode_stats"], stats)
        self.assertIsNotNone(fields["metrics"].speculative_decoding)
        build_metrics.assert_called_once_with(result)

    def _packed(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch is required for packed tensor regressions")
        namespace = {"torch": torch, "DEFAULT_PACKED_BUFFER_SIZE_BYTES": 8, "DEFAULT_PACKED_NUM_BUFFERS": 2}
        _load_definitions(
            self.sources["vllm-ascend"] / "vllm_ascend/distributed/weight_transfer/packed_tensor.py",
            {"packed_npu_ipc_producer", "packed_broadcast_producer"},
            namespace,
        )
        return torch, namespace

    def test_each_ipc_chunk_exports_a_fresh_reference_including_empty_tensors(self):
        torch, namespace = self._packed()
        exports = []

        def reduce_tensor(tensor):
            exports.append(tensor)
            return None, (len(exports),)

        namespace["reduce_tensor"] = reduce_tensor
        original_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)

        npu = SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None))
        for sizes in ((8, 1), (0,)):
            with self.subTest(sizes=sizes), patch.object(torch, "empty", cpu_empty), patch.object(
                torch, "npu", npu, create=True
            ):
                exports.clear()
                pairs = [(str(index), torch.ones(size, dtype=torch.uint8)) for index, size in enumerate(sizes)]
                chunks = list(namespace["packed_npu_ipc_producer"](iter(pairs), "npu", lambda item: item[1], 8))
                self.assertEqual(len(exports), len(chunks))
                self.assertEqual(
                    [chunk["ipc_handle"]["npu"] for chunk in chunks], [(index + 1,) for index in range(len(chunks))]
                )
                self.assertEqual(
                    [name for chunk in chunks for name in chunk["names"]], [name for name, tensor in pairs]
                )

    def test_packing_waits_for_the_callers_stream_before_broadcast(self):
        torch, namespace = self._packed()
        events = []
        source = object()

        class Stream:
            def synchronize(self):
                pass

            def wait_stream(self, stream):
                events.append(("wait", stream))

        npu = SimpleNamespace(Stream=Stream, current_stream=lambda: source, stream=contextlib.nullcontext)
        group = SimpleNamespace(broadcast=lambda *args, **kwargs: events.append(("broadcast", None)))
        namespace["_drain_packed_streams"] = lambda *args: None
        original_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)

        with patch.object(torch, "npu", npu, create=True), patch.object(torch, "empty", cpu_empty):
            namespace["packed_broadcast_producer"](
                iter([("weight", torch.ones(4, dtype=torch.uint8))]), group, 0, lambda item: item[1], 8, 2
            )
        self.assertEqual(events[:2], [("wait", source), ("broadcast", None)])


if __name__ == "__main__":
    unittest.main()

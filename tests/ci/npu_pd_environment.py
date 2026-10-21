"""Check PD dependencies before downloading models; no accelerator allocation."""

import argparse
import ctypes
import importlib
import importlib.metadata
import json


def check(encoder=False):
    from mooncake.engine import TransferEngine  # noqa: F401

    importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector")
    allocator = importlib.import_module("vllm_ascend.device_allocator.camem")
    if not allocator.camem_available:
        raise RuntimeError("PD sleep/wake needs the patched CaMem native allocator; run build_camem.py")
    versions = {}
    for name in ("mooncake-transfer-engine-npu", "vllm-ascend", "nixl", "nixl-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if encoder:
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError("ECCPUConnector needs the pinned host-memory NIXL/UCX dependencies")
        # Exercise DRAM registration, without pretending that NPU device-memory
        # NIXL or Ray's tensor-transport API has been implemented.
        storage = ctypes.create_string_buffer(4096)
        agent = NixlWrapper("vime-ec-preflight", nixl_agent_config(backends=["UCX"], capture_telemetry=True))
        descriptors = agent.get_reg_descs([(ctypes.addressof(storage), len(storage), 0, "")], "DRAM")
        agent.register_memory(descriptors, backends=["UCX"])
        agent.deregister_memory(descriptors)
    print(json.dumps({"npu_pd_dependencies": versions, "encoder_host_registration": encoder}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", action="store_true")
    check(**vars(parser.parse_args()))

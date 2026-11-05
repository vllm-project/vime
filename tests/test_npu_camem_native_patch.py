"""Compile the shipped allocator with a strict CPU-only CANN driver double."""

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_npu_pd_lifecycle_patch import PATCHES, new_side

NUM_GPUS = 0
REPO = Path(__file__).resolve().parents[1]

HEADER = r"""
#pragma once
#include <cstdlib>
#include <cstddef>
typedef int aclError;
typedef void* aclrtContext;
typedef void* aclrtStream;
typedef void* aclrtDrvMemHandle;
struct aclrtPhysicalMemProp {
  int handleType, allocationType, memAttr;
  struct { int id, type; } location;
  int reserve;
};
#define ACL_MEM_HANDLE_TYPE_NONE 0
#define ACL_MEM_ALLOCATION_TYPE_PINNED 0
#define ACL_HBM_MEM_HUGE 0
#define ACL_MEM_LOCATION_TYPE_DEVICE 0
#define ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM 0
#define ACL_ERROR_RT_MEMORY_ALLOCATION 207001
extern "C" {
int aclrtGetCurrentContext(aclrtContext*);
int aclrtCreateContext(aclrtContext*, int);
int aclrtSetCurrentContext(aclrtContext);
int aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp*, int, size_t*);
int aclrtReserveMemAddress(void**, size_t, int, void*, int);
int aclrtReleaseMemAddress(void*);
int aclrtMallocPhysical(aclrtDrvMemHandle*, size_t, aclrtPhysicalMemProp*, int);
int aclrtFreePhysical(aclrtDrvMemHandle);
int aclrtMapMem(void*, size_t, int, aclrtDrvMemHandle, int);
int aclrtUnmapMem(void*);
}
"""

DRIVER = r"""
#include "acl/acl.h"
#include <unordered_set>
static std::unordered_set<void*> mapped, handles, addresses;
static int unmap_error = 0, map_error = 0;
extern "C" {
int aclrtGetCurrentContext(aclrtContext* p) { *p = (void*)1; return 0; }
int aclrtCreateContext(aclrtContext* p, int) { *p = (void*)1; return 0; }
int aclrtSetCurrentContext(aclrtContext) { return 0; }
int aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp*, int, size_t* p) { *p = 64; return 0; }
int aclrtReserveMemAddress(void** p, size_t n, int, void*, int) {
  *p = malloc(n); addresses.insert(*p); return 0;
}
int aclrtReleaseMemAddress(void* p) {
  if (mapped.count(p) || !addresses.erase(p)) return 507899;
  free(p); return 0;
}
int aclrtMallocPhysical(aclrtDrvMemHandle* p, size_t n, aclrtPhysicalMemProp*, int) {
  *p = malloc(n); handles.insert(*p); return 0;
}
int aclrtFreePhysical(aclrtDrvMemHandle p) {
  if (!handles.erase(p)) return 507899;
  free(p); return 0;
}
int aclrtMapMem(void* p, size_t, int, aclrtDrvMemHandle h, int) {
  if (map_error) return map_error;
  return handles.count(h) && mapped.insert(p).second ? 0 : 507899;
}
int aclrtUnmapMem(void* p) {
  if (unmap_error) return unmap_error;
  return mapped.erase(p) ? 0 : 507899;
}
int live_handles() { return handles.size(); }
int live_mappings() { return mapped.size(); }
int live_addresses() { return addresses.size(); }
void fail_unmap(int code) { unmap_error = code; }
void fail_map(int code) { map_error = code; }
}
"""


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.skip("C++ compiler required for native allocator CPU tests")
    root = tmp_path_factory.mktemp("camem-driver")
    (root / "csrc").mkdir()
    (root / "vllm_ascend").mkdir()
    (root / "include/acl").mkdir(parents=True)
    (root / "lib64").mkdir()
    (root / "include/acl/acl.h").write_text(HEADER)
    source = new_side((PATCHES / "vllm-ascend.patch").read_text(), "csrc/camem_allocator.cpp")
    (root / "csrc/camem_allocator.cpp").write_text(source)
    driver_source = root / "driver.cpp"
    driver_source.write_text(DRIVER)
    library = root / "lib64/libascendcl.so"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-I" + str(root / "include"),
            str(driver_source),
            "-o",
            str(library),
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("camem_build", REPO / "docker/npu_patch/build_camem.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output = builder.build(root, root)
    modified = output.stat().st_mtime_ns
    assert builder.build(root, root).stat().st_mtime_ns == modified
    return output, library


def exercise(scenario, output, library):
    # The pytest process may have imported torch-npu, which loads real CANN
    # symbols globally. Isolate the mock driver so it can never allocate NPU.
    assert "torch" not in sys.modules
    spec = importlib.util.spec_from_file_location("vime_camem_allocator", output)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    assert native.MAPPING_STATE_VERSION == 1
    driver = ctypes.CDLL(str(library))
    allocator = ctypes.CDLL(str(output))
    allocator.my_malloc.argtypes = [ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    allocator.my_malloc.restype = ctypes.c_void_p
    allocator.my_free.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p]
    active = {}

    def malloc(handle):
        active[handle[2]] = handle

    def free(ptr):
        return active.pop(ptr)

    native.init_module(malloc, free)
    ptr = allocator.my_malloc(128, 0, None)
    handle = active[ptr]
    try:
        if scenario == "sleep-free":
            native.python_unmap_and_release(*handle)
            native.python_unmap_and_release(*handle)
            assert driver.live_handles() == driver.live_mappings() == 0
            allocator.my_free(ptr, 128, 0, None)
        elif scenario == "repeat-wake":
            for _ in range(3):
                native.python_unmap_and_release(*handle)
                native.python_create_and_map(*handle)
                native.python_create_and_map(*handle)
                assert driver.live_handles() == driver.live_mappings() == 1
        elif scenario == "unmap-error":
            driver.fail_unmap(507899)
            with pytest.raises(RuntimeError, match="aclrtUnmapMem.*507899.*device=0.*ptr=.*mapped=1"):
                native.python_unmap_and_release(*handle)
            assert driver.live_handles() == driver.live_mappings() == 1
            driver.fail_unmap(0)
            native.python_unmap_and_release(*handle)
        elif scenario in ("map-failure-free", "map-failure-retry"):
            native.python_unmap_and_release(*handle)
            driver.fail_map(507899)
            with pytest.raises(RuntimeError, match="aclrtMapMem.*507899.*mapped=0.*allocated=1"):
                native.python_create_and_map(*handle)
            assert driver.live_handles() == 1 and driver.live_mappings() == 0
            driver.fail_map(0)
            if scenario == "map-failure-retry":
                native.python_create_and_map(*handle)
                assert driver.live_handles() == driver.live_mappings() == 1
            else:
                native.python_unmap_and_release(*handle)
                assert driver.live_handles() == driver.live_mappings() == 0
        else:
            raise AssertionError(scenario)
    finally:
        driver.fail_map(0)
        driver.fail_unmap(0)
        if ptr in active:
            allocator.my_free(ptr, 128, 0, None)
        assert driver.live_handles() == driver.live_mappings() == driver.live_addresses() == 0


@pytest.mark.parametrize(
    "scenario", ["sleep-free", "repeat-wake", "unmap-error", "map-failure-free", "map-failure-retry"]
)
def test_native_mapping_lifecycle(compiled, scenario):
    env = {**os.environ, "LD_LIBRARY_PATH": str(compiled[1].parent)}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), scenario, *map(str, compiled)], env=env, check=True)


if __name__ == "__main__":
    exercise(*sys.argv[1:])

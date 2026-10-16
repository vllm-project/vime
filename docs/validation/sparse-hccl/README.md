# Native sparse HCCL container validation

These logs validate checkpoint-coordinate sparse HCCL synchronization in the VIME container.

- `patch-check.json`: both complete dependency patches apply to pristine pinned source; native sparse engine/helper match the recorded source revision verbatim.
- `patch-apply.log`: patches applied inside the existing VIME Docker container.
- `functional-manifest.json`: actual Python source and patch SHA256 values, installed package metadata, and successful reverse-apply checks against the patched trees.
- `imports.log`: real imports of the patched vLLM/Ascend trees and VIME updater.
- `vime-unit.log`: 29 passed, including actual four-rank HCCL gather and Bridge sparse/full bitwise reconstruction. This run predates the three newly added seed/API cases.
- `vime-native-cases.log`: four selected native cases passed (three additional cases plus one overlapping steady-update case); 17 cases deselected. Do not sum this count with the previous session as distinct tests.
- `vime-engine-cuda.log`: 70 CPU launch/API tests passed using their expected CUDA platform mapping. `vime-engine.log` retains the initial NPU-environment result (69 passed / one CUDA-backend expectation failed); this was a test-environment mismatch, not a production backend change.
- `ascend-unit.log`: 45 native sparse engine cases passed.
- `routed-unit.log`: 14 layout/expert cases passed, including temporary transpose-view address preservation and exception restoration.
- `receiver-oracle-final.log`: real TP2/PP2/EP2 case passed; five other file cases deselected. `receiver-oracle-final/.../validation.json` contains complete worker parameter hashes and generated tokens matched against native dense loading; `server.log` is the corresponding service output.
- `oracle-exit-code`: zero.

Timing note: pytest durations are entire session times, not weight synchronization latency. Absolute log timestamps retain the remote process clock.

Environment: host 192.168.13.190, container cjy-vime-0623, five selected NPUs 8–12 for the independent sender/receiver test. Complete pristine Python snapshots use the Dockerfile pins (vLLM e6bfe03a, Ascend fd815467) plus these patches. Installed native/CANN libraries are retained. Installed package version strings in the manifest describe existing metadata and must not be mistaken for the patched Python source revisions. The existing container is based on v0.21.0rc1-a3 and has been updated since creation; this is not a clean Dockerfile image build result.

30B full/sparse VIME measurements are still pending and will be published separately. No earlier direct-write performance result is attributed to these patches.

Published text logs normalize ANSI coloring, line endings and trailing whitespace only. `raw-log-sha256.json` identifies the original logs retained in the local evidence archive; event values and timestamps are unchanged.

"""CPU contracts for the S7 checkpoint test modes and patch ordering."""

import ast
import importlib.util
import shlex
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def glm_loader(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("VIME_TEST_GLM_MTP", raising=False)
    monkeypatch.delenv("VIME_TEST_GLM_EAGER", raising=False)

    def load():
        spec = importlib.util.spec_from_file_location("glm_npu_case", REPO / "tests/test_glm4.7_30B_A3B_npu.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def test_glm_g1_uses_native_non_colocate_without_mtp(glm_loader, monkeypatch):
    glm = glm_loader()
    launches = []
    monkeypatch.setattr(glm.U, "execute_train", lambda **kwargs: launches.append(kwargs))
    glm.execute()
    launch = launches[0]
    tokens = shlex.split(launch["train_args"])
    expected = {
        "--hf-checkpoint": glm.MODEL_DIR,
        "--ref-load": glm.MODEL_DIR,
        "--actor-num-gpus-per-node": "8",
        "--rollout-num-gpus": "8",
        "--rollout-num-gpus-per-engine": "4",
        "--tensor-model-parallel-size": "4",
        "--expert-model-parallel-size": "8",
        "--rope-type": "rope",
        "--vllm-additional-config": '{"weight_nz_mode":0}',
        "--num-rollout": "2",
    }
    for flag, value in expected.items():
        assert tokens[tokens.index(flag) + 1] == value
    assert "--vllm-enable-expert-parallel" in tokens
    assert "--ci-test" in tokens
    assert "--vllm-enforce-eager" not in tokens
    assert not {
        "--colocate",
        "--megatron-to-hf-mode",
        "--mtp-num-layers",
        "--enable-mtp-training",
        "--vllm-speculative-config",
        "--dspark-enabled",
    }.intersection(tokens)
    assert launch["num_gpus_per_node"] == 16
    assert launch["megatron_model_type"] == "glm4.7-30B-A3B"


@pytest.mark.parametrize("mtp,eager", [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")])
def test_glm_explicit_mtp_and_eager_modes(glm_loader, monkeypatch, mtp, eager):
    monkeypatch.setenv("VIME_TEST_GLM_MTP", mtp)
    monkeypatch.setenv("VIME_TEST_GLM_EAGER", eager)
    glm = glm_loader()
    launches = []
    monkeypatch.setattr(glm.U, "execute_train", lambda **kwargs: launches.append(kwargs))
    glm.execute()
    tokens = shlex.split(launches[0]["train_args"])
    assert ("--vllm-enforce-eager" in tokens) == (eager == "1")
    for flag in (
        "--mtp-num-layers",
        "--enable-mtp-training",
        "--mtp-loss-scaling-factor",
        "--vllm-speculative-config",
    ):
        assert (flag in tokens) == (mtp == "1")
    if mtp == "1":
        assert tokens[tokens.index("--mtp-num-layers") + 1] == "1"
        assert tokens[tokens.index("--mtp-loss-scaling-factor") + 1] == "0.2"
        assert tokens[tokens.index("--vllm-speculative-config") + 1] == '{"method":"mtp","num_speculative_tokens":1}'
    assert "--colocate" not in tokens
    assert "--dspark-enabled" not in tokens
    assert tokens[tokens.index("--actor-num-gpus-per-node") + 1] == "8"
    assert tokens[tokens.index("--rollout-num-gpus") + 1] == "8"
    assert tokens[tokens.index("--rollout-num-gpus-per-engine") + 1] == "4"


def test_glm_prepare_preserves_ci_download_defaults(glm_loader, monkeypatch):
    glm = glm_loader()
    commands = []
    monkeypatch.setattr(glm.U, "exec_command", commands.append)
    glm.prepare()
    assert commands == [
        f"mkdir -p {shlex.quote(f'{glm.TEST_ROOT}/models')} {shlex.quote(f'{glm.TEST_ROOT}/datasets')}",
        f"hf download zai-org/GLM-4.7-Flash --local-dir {shlex.quote(glm.MODEL_DIR)}",
        "hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {shlex.quote(glm.DATASET_DIR)}",
    ]


def test_glm_shell_keeps_g1_model_and_serving_configuration():
    script = (REPO / "scripts/run-glm4.7-30B-A3B-npu.sh").read_text()
    assert 'source "${SCRIPT_DIR}/models/glm4.7-30B-A3B.sh"' in script
    assert "--rope-type rope" in script
    assert "--vllm-additional-config '{\"weight_nz_mode\":0}'" in script
    assert "--vllm-enable-expert-parallel" in script
    for flag in (
        "--colocate",
        "--megatron-to-hf-mode",
        "--mtp-num-layers",
        "--enable-mtp-training",
        "--vllm-speculative-config",
    ):
        assert flag not in script


@pytest.fixture
def qwen30(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "models").mkdir()
    spec = importlib.util.spec_from_file_location("qwen30_npu_case", REPO / "tests/test_qwen3_30B_A3B_npu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_30b_hf_mode_skips_conversion(qwen30, monkeypatch):
    commands = []
    launches = []
    monkeypatch.setattr(qwen30.U, "exec_command", commands.append)
    monkeypatch.setattr(qwen30.U, "execute_train", lambda **kwargs: launches.append(kwargs))
    assert qwen30.prepare(torch_dist_ref_load=False) is None
    qwen30.execute()
    assert not any("torch.distributed.run" in cmd or "rm -rf" in cmd for cmd in commands)
    args = launches[0]["train_args"]
    assert f"--ref-load {shlex.quote(qwen30.MODEL_DIR)} " in args
    assert "--colocate " in args
    assert "--tensor-model-parallel-size 4 " in args
    assert "--expert-model-parallel-size 8 " in args
    assert "weight_nz_mode" in args


def test_default_torch_dist_mode_uses_new_output_and_ref_load(qwen30, monkeypatch, tmp_path):
    commands = []
    launches = []
    existing = tmp_path / "models/Qwen3-30B-A3B_torch_dist"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("existing checkpoint")

    def execute(command):
        commands.append(command)
        if "torch.distributed.run" in command:
            tokens = shlex.split(command)
            target = Path(tokens[tokens.index("--save") + 1])
            (target / "latest_checkpointed_iteration.txt").write_text("release")
            (target / ".metadata").write_bytes(b"test fixture")

    monkeypatch.setattr(qwen30.U, "exec_command", execute)
    monkeypatch.setattr(qwen30.U, "execute_train", lambda **kwargs: launches.append(kwargs))
    checkpoint = qwen30.prepare()
    qwen30.execute(checkpoint)
    assert Path(checkpoint) != existing
    assert sentinel.read_text() == "existing checkpoint"
    assert not any("rm -rf" in command for command in commands)
    conversion = next(command for command in commands if "torch.distributed.run" in command)
    assert "VIME_PLATFORM=npu" in conversion
    assert "--nproc-per-node 8 " in conversion
    args = launches[0]["train_args"]
    assert f"--ref-load {shlex.quote(checkpoint)} " in args
    assert "--load " not in args
    assert "--colocate " in args
    assert "weight_nz_mode" in args


@pytest.mark.parametrize("override,enabled", [(None, True), ("1", True), ("0", False)])
def test_30b_main_checkpoint_mode(qwen30, monkeypatch, override, enabled):
    monkeypatch.delenv("VIME_TEST_TORCH_DIST_REF_LOAD", raising=False)
    if override is not None:
        monkeypatch.setenv("VIME_TEST_TORCH_DIST_REF_LOAD", override)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(key, raising=False)
    checkpoint = object()
    launches = []

    def prepare(*, torch_dist_ref_load):
        assert torch_dist_ref_load is enabled
        return checkpoint if enabled else None

    monkeypatch.setattr(qwen30, "prepare", prepare)
    monkeypatch.setattr(qwen30, "execute", launches.append)
    qwen30.main()
    assert launches == [checkpoint if enabled else None]


def test_converter_bootstraps_before_first_megatron_import():
    tree = ast.parse((REPO / "tools/convert_hf_to_torch_dist.py").read_text())
    first_megatron = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("megatron.")
    )
    bootstrap = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "vime.backends.megatron_utils" for alias in node.names)
    )
    assert bootstrap < first_megatron
    assert "vime.utils.common" not in ast.unparse(tree)


def test_common_megatron_patch_is_snapshotted_before_npu_patch():
    entries = [
        line.split("|")
        for line in (REPO / "docker/npu_patch/series.conf").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    megatron = [entry for entry in entries if entry[0] == "/root/Megatron-LM"]
    assert megatron == [
        ["/root/Megatron-LM", "megatron-common.patch", "docker/patch/latest/megatron.patch"],
        ["/root/Megatron-LM", "megatron.patch", "docker/npu_patch/megatron.patch"],
    ]
    dockerfile = (REPO / "docker/Dockerfile.npu").read_text()
    assert "COPY docker/patch/latest/megatron.patch /opt/npu_patch/megatron-common.patch" in dockerfile
    assert "/opt/vime_patch/megatron.patch" not in dockerfile


def _megatron_patch_additions(path, patch_path="docker/npu_patch/megatron.patch"):
    patch = (REPO / patch_path).read_text()
    section = patch.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("diff --git ", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))


def test_npu_patch_keeps_public_transformer_layer():
    patch = (REPO / "docker/npu_patch/megatron.patch").read_text()
    assert "diff --git a/megatron/core/transformer/transformer_layer.py " not in patch


def test_post_layernorm_flags_remain_dataclass_generated():
    additions = _megatron_patch_additions(
        "megatron/core/transformer/transformer_config.py", "docker/patch/latest/megatron.patch"
    )
    fields = ast.parse(textwrap.dedent(additions)).body
    defaults = {node.target.id: ast.literal_eval(node.value) for node in fields if isinstance(node, ast.AnnAssign)}
    npu_patch = (REPO / "docker/npu_patch/megatron.patch").read_text()
    for name in ("post_self_attn_layernorm", "post_mlp_layernorm"):
        assert defaults[name] is False
        assert f"--{name.replace('_', '-')}" not in npu_patch

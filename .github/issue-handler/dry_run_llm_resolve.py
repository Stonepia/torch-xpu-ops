"""Dry-run: LLM-based test path resolution via opencode.

Tests the approach against 6 known auto-verified issues.
Measures LLM call latency for each.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

PYTORCH_DIR = Path.home() / "pytorch"
OPENCODE_CMD = "opencode"

# ---------------------------------------------------------------------------
# Training data: 6 auto-verified issues
# ---------------------------------------------------------------------------
CASES = [
    {
        "number": 1963,
        "raw_ref": 'pytest -v test_ops.py -k "test_fake_autocast_linalg_pinv_xpu_float32 or test_fake_autocast_pinverse_xpu_float32 or test_fake_crossref_backward_amp_nn_functional_bilinear_xpu_float32"',
        "expected": "CANNOT_VERIFY",
        "notes": "bare test_ops.py, -k filter matches 0 dynamic tests",
    },
    {
        "number": 2283,
        "raw_ref": 'pytest -v "test/xpu/test_sparse_xpu.py::TestSparseAnyXPU::test_gradcheck_mm_SparseCSR_masked_fast_xpu_complex128" "test/xpu/test_sparse_csr_xpu.py::TestSparseCSRXPU::test_sampled_addmm_xpu_float64"',
        "expected": "RESOLVE",
        "notes": "test/xpu/ needs third_party/torch-xpu-ops/ prefix",
    },
    {
        "number": 2891,
        "raw_ref": "python test/inductor/test_cuda_repro.py CudaReproTests.test_effn_attn_bias_padding",
        "expected": "RESOLVE",
        "notes": "python invocation, file exists as-is",
    },
    {
        "number": 1951,
        "raw_ref": (
            "op_ut,third_party.torch-xpu-ops.test.xpu.test_ops_xpu.TestCommonXPU,"
            "test_out_triangular_solve_xpu_float32\n"
            "op_ut,third_party.torch-xpu-ops.test.xpu.test_ops_xpu.TestCommonXPU,"
            "test_out_cholesky_inverse_xpu_float32"
        ),
        "expected": "RESOLVE",
        "notes": "CI metadata format, dots → slashes",
    },
    {
        "number": 2800,
        "raw_ref": "python test/test_scaled_matmul_cuda.py TestFP8MatmulXPU.test_scaled_mm_vs_emulated_float32_x_cm_True_y_cm_True_xpu",
        "expected": "RESOLVE",
        "notes": "python invocation, file exists as-is",
    },
    {
        "number": 2295,
        "raw_ref": (
            'pytest -v "test/nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::'
            'test_embedding_bag_device_xpu_int32_int32_float64" '
            '"test/test_reductions.py::TestReductionsXPU::test_argminmax_multiple_xpu_float64"'
        ),
        "expected": "RESOLVE",
        "notes": "upstream pytorch test paths, files exist",
    },
]


# ---------------------------------------------------------------------------
# Gather filesystem context for the LLM
# ---------------------------------------------------------------------------
def _find_candidates(raw_ref: str) -> str:
    """Extract filenames from raw_ref and find matching files on disk."""
    # Extract .py filenames
    filenames = set()
    for m in re.finditer(r'(test_\w+\.py|test\w+\.py)', raw_ref):
        filenames.add(m.group(1))
    # Also try extracting from CI metadata (dots → filename)
    for m in re.finditer(r'(\w+\.py)', raw_ref):
        filenames.add(m.group(1))

    if not filenames:
        return "(no candidate files found)"

    results = []
    for fn in sorted(filenames):
        r = subprocess.run(
            f"find {PYTORCH_DIR} -name '{fn}' -not -path '*__pycache__*' -not -path '*/build/*' -not -path '*/.git/*' 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.strip().splitlines():
            if line:
                rel = str(Path(line).relative_to(PYTORCH_DIR))
                results.append(rel)

    return "\n".join(results) if results else "(no matching files on disk)"


# ---------------------------------------------------------------------------
# LLM call via opencode
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
You are a test path resolver for the PyTorch XPU CI pipeline.
Your job: map a CI test reference to a runnable local command.

## Test reference from CI issue:
```
{raw_ref}
```

## Matching test files on disk (paths relative to ~/pytorch/):
```
{candidates}
```

## Rules:
- Working directory is ~/pytorch/ — all paths must be relative to it
- XPU tests from torch-xpu-ops are at third_party/torch-xpu-ops/test/xpu/ under ~/pytorch/
- If the reference says test/xpu/foo.py, the actual file is third_party/torch-xpu-ops/test/xpu/foo.py
- Dynamic test classes: TestFooXPU is generated from TestFoo via instantiate_device_type_tests — the class IS valid even though grep won't find "class TestFooXPU" literally
- For CI metadata format (op_ut,module.path.TestClass,method): convert dots to slashes for the file path, keep class and method as pytest selectors
- For pytest commands: use pytest -xvs with "file.py::Class::method" paths
- For python unittest commands (python file.py Class.method): keep the format

## Output:
Return ONLY the runnable command on a single line. No explanation, no markdown.
If you genuinely cannot resolve the path, return exactly: CANNOT_RESOLVE
"""


def _llm_resolve(raw_ref: str) -> tuple[str | None, float]:
    """Call opencode to resolve test path. Returns (command, elapsed_seconds)."""
    candidates = _find_candidates(raw_ref)
    prompt = PROMPT_TEMPLATE.format(raw_ref=raw_ref, candidates=candidates)

    cmd = [
        OPENCODE_CMD, "run", "--format", "json",
        "--dir", str(PYTORCH_DIR),
        "--dangerously-skip-permissions",
        prompt,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=90,
            stdin=subprocess.DEVNULL,
        )
        elapsed = time.time() - t0

        # Parse opencode JSON event stream
        text_parts = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                part = event.get("part", {})
                text_parts.append(part.get("text", ""))

        response = "".join(text_parts).strip()
        # Clean up: remove markdown fences if LLM wrapped it
        response = re.sub(r'^```\w*\n', '', response)
        response = re.sub(r'\n```$', '', response)
        response = response.strip()

        if not response or "CANNOT_RESOLVE" in response:
            return None, elapsed
        return response, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return None, elapsed


# ---------------------------------------------------------------------------
# Validate resolved command
# ---------------------------------------------------------------------------
def _validate_command(cmd: str) -> tuple[bool, str]:
    """Check that the file paths in the command exist on disk."""
    # Extract file paths
    paths = []
    for m in re.finditer(r'(?:"|^|\s)(\S+\.py)', cmd):
        p = m.group(1).strip('"').split("::")[0]
        if p not in ("pytest",):
            paths.append(p)

    if not paths:
        return False, "no file paths found in command"

    missing = []
    for p in paths:
        full = PYTORCH_DIR / p
        if not full.exists():
            missing.append(p)

    if missing:
        return False, f"missing files: {missing}"
    return True, f"all {len(paths)} file(s) exist"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DRY RUN: LLM-based test path resolution via opencode")
    print("=" * 70)

    total_time = 0.0

    for case in CASES:
        n = case["number"]
        print(f"\n{'─' * 70}")
        print(f"#{n}: {case['notes']}")
        print(f"  Raw ref:    {case['raw_ref'][:100]}...")
        print(f"  Expected:   {case['expected']}")

        # Find candidates (filesystem)
        t0 = time.time()
        candidates = _find_candidates(case["raw_ref"])
        find_time = time.time() - t0
        print(f"  Candidates: ({find_time:.1f}s to find)")
        for c in candidates.split("\n")[:5]:
            print(f"    {c}")
        if candidates.count("\n") > 4:
            print(f"    ... and {candidates.count(chr(10)) - 4} more")

        # LLM resolve
        print(f"  Calling opencode...")
        resolved, elapsed = _llm_resolve(case["raw_ref"])
        total_time += elapsed
        print(f"  LLM time:   {elapsed:.1f}s")
        print(f"  Resolved:   {(resolved or 'CANNOT_RESOLVE')[:120]}")

        # Validate
        if resolved:
            valid, detail = _validate_command(resolved)
            print(f"  Valid:      {'✅' if valid else '❌'} {detail}")
        else:
            print(f"  Valid:      ⚠️  no command to validate")

    print(f"\n{'=' * 70}")
    print(f"TOTAL LLM TIME: {total_time:.1f}s across {len(CASES)} cases")
    print(f"AVERAGE:         {total_time / len(CASES):.1f}s per call")
    print("=" * 70)


if __name__ == "__main__":
    main()

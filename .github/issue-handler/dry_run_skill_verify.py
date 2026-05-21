"""Dry-run: test the new verify_existence.py + test-verification skill
against 6 known auto-verified issues.

Does NOT modify any issues — just prints what would happen.
Measures LLM call time per issue.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Add issue_handler to path so we can reuse shared utils
sys.path.insert(0, str(Path(__file__).parent))
from issue_handler.utils.xpu_env import ensure_xpu_ready

PYTORCH_DIR = Path.home() / "pytorch"
OPENCODE_CMD = "opencode"
SKILLS_DIR = Path.home() / "torch-xpu-ops/.github/skills"

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
CASES = [
    {
        "number": 1963,
        "raw_ref": 'pytest -v test_ops.py -k "test_fake_autocast_linalg_pinv_xpu_float32 or test_fake_autocast_pinverse_xpu_float32 or test_fake_crossref_backward_amp_nn_functional_bilinear_xpu_float32"',
        "title": "test_fake_autocast failures in test_ops.py",
        "expected_status": "CANNOT_VERIFY",
        "notes": "bare test_ops.py, -k filter matches 0 dynamic tests",
    },
    {
        "number": 2283,
        "raw_ref": 'pytest -v "test/xpu/test_sparse_xpu.py::TestSparseAnyXPU::test_gradcheck_mm_SparseCSR_masked_fast_xpu_complex128" "test/xpu/test_sparse_csr_xpu.py::TestSparseCSRXPU::test_sampled_addmm_xpu_float64"',
        "title": "sparse XPU test failures in test_sparse_xpu.py",
        "expected_status": "FAILED",
        "notes": "test/xpu/ needs third_party/torch-xpu-ops/ prefix",
    },
    {
        "number": 2891,
        "raw_ref": "python test/inductor/test_cuda_repro.py CudaReproTests.test_effn_attn_bias_padding",
        "title": "inductor test_effn_attn_bias_padding failure",
        "expected_status": "PASSED_OR_FAILED",
        "notes": "python invocation, file exists as-is, depends on current state",
    },
    {
        "number": 1951,
        "raw_ref": (
            "op_ut,third_party.torch-xpu-ops.test.xpu.test_ops_xpu.TestCommonXPU,"
            "test_out_triangular_solve_xpu_float32\n"
            "op_ut,third_party.torch-xpu-ops.test.xpu.test_ops_xpu.TestCommonXPU,"
            "test_out_cholesky_inverse_xpu_float32"
        ),
        "title": "test_out_triangular_solve and cholesky_inverse XPU failures",
        "expected_status": "FAILED",
        "notes": "CI metadata format, dots -> slashes",
    },
    {
        "number": 2800,
        "raw_ref": "python test/test_scaled_matmul_cuda.py TestFP8MatmulXPU.test_scaled_mm_vs_emulated_float32_x_cm_True_y_cm_True_xpu",
        "title": "test_scaled_mm_vs_emulated XPU failure",
        "expected_status": "PASSED_OR_FAILED",
        "notes": "python invocation, file exists as-is, depends on current state",
    },
    {
        "number": 2295,
        "raw_ref": (
            'pytest -v "test/nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::'
            'test_embedding_bag_device_xpu_int32_int32_float64" '
            '"test/test_reductions.py::TestReductionsXPU::test_argminmax_multiple_xpu_float64"'
        ),
        "title": "embedding and reduction XPU test failures",
        "expected_status": "FAILED",
        "notes": "upstream pytorch test paths, files exist",
    },
]


# ---------------------------------------------------------------------------
# Build prompt (same as verify_existence.py)
# ---------------------------------------------------------------------------
def _build_prompt(case: dict) -> str:
    return (
        f"Verify whether issue #{case['number']} still reproduces.\n\n"
        f"## Issue Title\n{case['title']}\n\n"
        f"## Raw Test Reference\n```\n{case['raw_ref']}\n```\n\n"
        f"Follow the test-verification skill instructions. "
        f"Resolve the path, run the test, and output the JSON result."
    )


# ---------------------------------------------------------------------------
# Run opencode
# ---------------------------------------------------------------------------
def _run_opencode(prompt: str) -> tuple[str, float]:
    """Run opencode and return (parsed_text_output, elapsed_seconds)."""
    skills_hint = (
        f"\n\n## Context\n"
        f"XPU agent skills and instructions are in {SKILLS_DIR} "
        f"and {SKILLS_DIR.parent / 'instructions'}. "
        f"Read the relevant SKILL.md before starting work."
        f"\nThe skill for this task is: {SKILLS_DIR / 'test-verification' / 'SKILL.md'}"
    )
    full_prompt = prompt + skills_hint

    cmd = [
        OPENCODE_CMD, "run", "--format", "json",
        "--dir", str(PYTORCH_DIR),
        "--dangerously-skip-permissions",
        full_prompt,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            stdin=subprocess.DEVNULL,
        )
        elapsed = time.time() - t0

        # Parse opencode JSON events
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

        return "".join(text_parts), elapsed
    except subprocess.TimeoutExpired:
        return "", time.time() - t0


def _parse_result(output: str) -> dict | None:
    """Parse JSON from agent output."""
    json_blocks = re.findall(r'```json\s*\n(.*?)```', output, re.DOTALL)
    if json_blocks:
        raw = json_blocks[-1].strip()
    else:
        matches = re.findall(r'\{[^{}]*"status"[^{}]*\}', output, re.DOTALL)
        if matches:
            raw = matches[-1]
        else:
            return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DRY RUN: test-verification skill via opencode")
    print("=" * 70)

    # Pre-flight: ensure XPU environment is ready
    print("\n🔧 Pre-flight: checking XPU environment...")
    if not ensure_xpu_ready():
        print("❌ FATAL: Cannot get XPU environment ready. Aborting.")
        return
    print("✅ XPU environment ready.\n")

    total_time = 0.0
    results = []

    for case in CASES:
        n = case["number"]
        print(f"\n{'─' * 70}")
        print(f"#{n}: {case['notes']}")
        print(f"  Raw ref:    {case['raw_ref'][:100]}...")
        print(f"  Expected:   {case['expected_status']}")

        prompt = _build_prompt(case)
        print(f"  Running opencode...")

        output, elapsed = _run_opencode(prompt)
        total_time += elapsed
        print(f"  Time:       {elapsed:.1f}s")

        result = _parse_result(output)
        if result:
            status = result.get("status", "UNKNOWN")
            refined = result.get("refined_command", "N/A")
            reason = result.get("reason", "N/A")
            print(f"  Status:     {status}")
            print(f"  Refined:    {refined[:120]}")
            print(f"  Reason:     {reason}")

            # Check expected
            expected = case["expected_status"]
            if expected == "PASSED_OR_FAILED":
                match = status in ("PASSED", "FAILED")
            else:
                match = status == expected
            print(f"  Match:      {'✅' if match else '❌'} (expected {expected})")
            results.append({"number": n, "status": status, "expected": expected,
                            "match": match, "time": elapsed, "refined": refined})
        else:
            print(f"  Status:     ❌ PARSE_FAILED")
            print(f"  Raw output (last 500): {output[-500:]}")
            results.append({"number": n, "status": "PARSE_FAILED", "expected": case["expected_status"],
                            "match": False, "time": elapsed, "refined": ""})

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'#':<8} {'Status':<18} {'Expected':<18} {'Match':<6} {'Time':>6} Refined Command")
    print(f"{'─' * 70}")
    for r in results:
        m = "✅" if r["match"] else "❌"
        print(f"#{r['number']:<7} {r['status']:<18} {r['expected']:<18} {m:<6} {r['time']:>5.1f}s {r['refined'][:40]}")
    print(f"{'─' * 70}")
    print(f"TOTAL TIME: {total_time:.1f}s | AVG: {total_time/len(CASES):.1f}s/call")
    matched = sum(1 for r in results if r["match"])
    print(f"ACCURACY: {matched}/{len(results)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

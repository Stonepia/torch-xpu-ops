"""Dry-run: test new path resolution logic against known DONE issues.

Compares old _fix_test_path behavior vs new _resolve_test_path (filesystem-based).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

PYTORCH_DIR = Path.home() / "pytorch"

# ---------------------------------------------------------------------------
# Training data: 6 auto-verified issues
# ---------------------------------------------------------------------------
CASES = [
    {
        "number": 1963,
        "reproducer": 'pytest -v test_ops.py -k "test_fake_autocast_linalg_pinv_xpu_float32 or test_fake_autocast_pinverse_xpu_float32 or test_fake_crossref_backward_amp_nn_functional_bilinear_xpu_float32"',
        "expected_verdict": "CANNOT_VERIFY",  # 0 items collected, was false DONE
        "notes": "test_ops.py path not resolved; even with correct path, -k filter matches 0 dynamic tests",
    },
    {
        "number": 2283,
        "reproducer": 'pytest -v "test/xpu/test_sparse_xpu.py::TestSparseAnyXPU::test_gradcheck_mm_SparseCSR_masked_fast_xpu_complex128" "test/xpu/test_sparse_csr_xpu.py::TestSparseCSRXPU::test_sampled_addmm_xpu_float64"',
        "expected_verdict": "CANNOT_VERIFY",  # file exists but class TestSparseAnyXPU doesn't
        "notes": "test/xpu/ rewritten to third_party/torch-xpu-ops/test/xpu/ — file exists but TestSparseAnyXPU not in it",
    },
    {
        "number": 2891,
        "reproducer": "python test/inductor/test_cuda_repro.py CudaReproTests.test_effn_attn_bias_padding",
        "expected_verdict": "FIXED",  # Ran 1 test, OK — legit
        "notes": "Legit pass",
    },
    {
        "number": 1951,
        "reproducer_from_failed_tests": [
            "third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_triangular_solve_xpu_float32",
            "third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestCommonXPU::test_out_cholesky_inverse_xpu_float32",
        ],
        "expected_verdict": "STILL_FAILING",  # 5 xfailed
        "notes": "CI metadata reproducer rejected; cmd built from Failed Tests. All xfailed = bug still exists",
    },
    {
        "number": 2800,
        "reproducer": "python test/test_scaled_matmul_cuda.py TestFP8MatmulXPU.test_scaled_mm_vs_emulated_float32_x_cm_True_y_cm_True_xpu",
        "expected_verdict": "FIXED",  # Ran 1 test, OK — legit
        "notes": "Legit pass",
    },
    {
        "number": 2295,
        "reproducer": 'pytest -v "test/nn/test_embedding.py::TestEmbeddingNNDeviceTypeXPU::test_embedding_bag_device_xpu_int32_int32_float64" "test/test_reductions.py::TestReductionsXPU::test_argminmax_multiple_xpu_float64"',
        "expected_verdict": "FIXED",  # Ran 1 test OK, but suspicious (7 tests → 1 ran)
        "notes": "Suspicious — reproducer had 7 tests but only 1 ran",
    },
]


# ---------------------------------------------------------------------------
# OLD logic (current code)
# ---------------------------------------------------------------------------
def _fix_test_path_OLD(test_path: str) -> str:
    parts = test_path.split("::", 1)
    file_part = parts[0]
    rest = f"::{parts[1]}" if len(parts) > 1 else ""
    if "/" in file_part:
        if file_part.startswith("test/xpu/"):
            return f"third_party/torch-xpu-ops/{file_part}{rest}"
        return test_path
    if file_part.endswith("_xpu.py"):
        return f"third_party/torch-xpu-ops/test/xpu/{file_part}{rest}"
    if file_part.startswith("test_") and file_part.endswith(".py"):
        return f"test/{file_part}{rest}"
    return test_path


# ---------------------------------------------------------------------------
# NEW logic: filesystem-based resolution
# ---------------------------------------------------------------------------
def _resolve_test_path(test_path: str, issue_body: str = "") -> str | None:
    """Resolve a test path to an actual file on disk.

    Returns the resolved path relative to PYTORCH_DIR, or None if not found.
    """
    parts = test_path.split("::", 1)
    file_part = parts[0].strip().strip('"').strip("'")
    rest = f"::{parts[1]}" if len(parts) > 1 else ""

    # If already absolute or has a known prefix, just check existence
    if "/" in file_part:
        # Apply known prefix rewrites
        if file_part.startswith("test/xpu/"):
            candidate = f"third_party/torch-xpu-ops/{file_part}"
        else:
            candidate = file_part

        full = PYTORCH_DIR / candidate
        if full.exists():
            # Validate test class exists in file if specified
            if rest:
                class_name = rest.split("::")[1] if "::" in rest[2:] else None
                if class_name and not _class_in_file(full, class_name):
                    return None  # class doesn't exist in this file
            return f"{candidate}{rest}"
        return None

    # Bare filename — search the filesystem
    filename = file_part
    if not filename.endswith(".py"):
        return test_path  # not a python file, return as-is

    # Search for the file
    candidates = list(PYTORCH_DIR.rglob(filename))
    # Filter out __pycache__, build dirs, .git
    candidates = [
        c for c in candidates
        if "__pycache__" not in str(c)
        and "/build/" not in str(c)
        and "/.git/" not in str(c)
    ]

    if not candidates:
        return None

    # If we have a test class in rest, validate it exists
    class_name = None
    if rest:
        # rest is like "::TestFoo::test_bar" — extract TestFoo
        rest_parts = rest.lstrip(":").split("::")
        for rp in rest_parts:
            if rp.startswith("Test"):
                class_name = rp
                break

    if class_name:
        valid = [c for c in candidates if _class_in_file(c, class_name)]
        if valid:
            candidates = valid
        else:
            return None  # class doesn't exist in any matching file

    if len(candidates) == 1:
        return str(candidates[0].relative_to(PYTORCH_DIR)) + rest

    # Multiple matches — disambiguate
    is_xpu = bool(re.search(r'xpu|XPU', issue_body))
    for c in candidates:
        rel = str(c.relative_to(PYTORCH_DIR))
        if is_xpu and "torch-xpu-ops" in rel:
            return rel + rest
        if not is_xpu and rel.startswith("test/"):
            return rel + rest

    # Fallback to first
    return str(candidates[0].relative_to(PYTORCH_DIR)) + rest


def _class_in_file(filepath: Path, class_name: str) -> bool:
    """Check if a test class is defined or generated in a file."""
    try:
        content = filepath.read_text(errors="ignore")
        # Direct class definition
        if f"class {class_name}" in content:
            return True
        # instantiate_device_type_tests generates classes like TestFooXPU from TestFoo
        # Check if base class exists (strip device suffix)
        base = re.sub(r'(XPU|CUDA|CPU|Meta)$', '', class_name)
        if base != class_name and f"class {base}" in content:
            return True
        # Also check for instantiate_device_type_tests call
        if "instantiate_device_type_tests" in content and base != class_name:
            return True
        return False
    except Exception:
        return False


def _resolve_pytest_cmd(cmd: str, issue_body: str = "") -> tuple[str | None, list[str]]:
    """Resolve all test paths in a pytest command.

    Returns (resolved_cmd, list_of_warnings).
    Returns (None, warnings) if any path can't be resolved.
    """
    warnings = []

    # Case 1: pytest with -k flag and bare filename
    # e.g., pytest -v test_ops.py -k "test_foo"
    m = re.match(r'(pytest\s+(?:-\S+\s+)*)(\S+\.py)\s+(.*)', cmd)
    if m:
        prefix, filepath, suffix = m.groups()
        resolved = _resolve_test_path(filepath, issue_body)
        if resolved is None:
            warnings.append(f"Cannot resolve path: {filepath}")
            return None, warnings
        return f"{prefix}{resolved} {suffix}", warnings

    # Case 2: pytest with explicit test paths (quoted or not)
    # e.g., pytest -v "test/xpu/foo.py::Class::method" "test/xpu/bar.py::..."
    test_paths = re.findall(r'"([^"]+\.py(?:::\S*)?)"', cmd)
    if not test_paths:
        # Try unquoted
        test_paths = re.findall(r'(?:^|\s)(\S+\.py(?:::\S*)?)', cmd)
        # Filter out pytest itself
        test_paths = [t for t in test_paths if not t.startswith("-") and t != "pytest"]

    if test_paths:
        resolved_paths = []
        for tp in test_paths:
            resolved = _resolve_test_path(tp, issue_body)
            if resolved is None:
                warnings.append(f"Cannot resolve path: {tp}")
            else:
                resolved_paths.append(resolved)

        if not resolved_paths:
            return None, warnings

        if len(resolved_paths) < len(test_paths):
            warnings.append(
                f"Only {len(resolved_paths)}/{len(test_paths)} paths resolved"
            )

        # Reconstruct: extract flags from original cmd
        flags = re.findall(r'-\S+', cmd.split(test_paths[0])[0])
        flag_str = " ".join(flags) if flags else "-v"
        path_str = " ".join(f'"{p}"' for p in resolved_paths)
        return f"pytest {flag_str} {path_str}", warnings

    # Case 3: python script.py args — just validate file exists, keep cmd intact
    m = re.match(r'(python\d?\s+)(\S+\.py)(.*)', cmd)
    if m:
        prefix, filepath, suffix = m.groups()
        full = PYTORCH_DIR / filepath
        if full.exists():
            return cmd, warnings  # keep original command as-is
        # Try searching
        candidates = list(PYTORCH_DIR.rglob(Path(filepath).name))
        candidates = [c for c in candidates if "__pycache__" not in str(c) and "/build/" not in str(c)]
        if candidates:
            resolved = str(candidates[0].relative_to(PYTORCH_DIR))
            return f"{prefix}{resolved}{suffix}", warnings
        warnings.append(f"Cannot resolve path: {filepath}")
        return None, warnings

    return cmd, warnings  # can't parse, return as-is


# ---------------------------------------------------------------------------
# Pre-check: --collect-only validation
# ---------------------------------------------------------------------------
def _precheck_collect(cmd: str, workdir: Path = PYTORCH_DIR) -> tuple[bool, int, str]:
    """Run pytest --collect-only to verify tests exist before full run.

    Returns (ok, count, output).
    ok=True means count > 0 tests would be collected.
    """
    # Only works for pytest commands
    if not cmd.strip().startswith("pytest"):
        return True, -1, "not a pytest cmd, skip precheck"

    collect_cmd = cmd + " --collect-only -q"
    env_setup = (
        "source ~/intel/oneapi/setvars.sh --force 2>/dev/null; "
        "source ~/pytorch/.venv/bin/activate; "
    )
    try:
        result = subprocess.run(
            env_setup + collect_cmd, cwd=str(workdir),
            capture_output=True, text=True,
            timeout=120, shell=True, executable="/bin/bash",
        )
        output = result.stdout + result.stderr
        # Count collected tests
        m = re.search(r"(\d+) tests? collected", output)
        if m:
            count = int(m.group(1))
            return count > 0, count, output[-500:]
        # "no tests ran" or "collected 0 items"
        if "collected 0 items" in output or "no tests ran" in output:
            return False, 0, output[-500:]
        # If rc != 0 and no collection info, likely error
        if result.returncode != 0:
            return False, 0, output[-500:]
        return True, -1, output[-500:]  # can't determine, assume ok
    except subprocess.TimeoutExpired:
        return False, 0, "collect-only timed out"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DRY RUN: old vs new path resolution")
    print("=" * 70)

    for case in CASES:
        n = case["number"]
        print(f"\n{'─' * 70}")
        print(f"#{n}: {case['notes']}")

        if "reproducer" in case:
            cmd = case["reproducer"]
            print(f"  Reproducer:     {cmd[:120]}")

            # Old behavior
            old_cmd = re.sub(
                r'(?<=["\s])test_\S+\.py(?:::\S+)?|^test_\S+\.py(?:::\S+)?',
                lambda m: _fix_test_path_OLD(m.group(0)), cmd
            )
            print(f"  Old resolved:   {old_cmd[:120]}")

            # New behavior
            new_cmd, warns = _resolve_pytest_cmd(cmd, f"xpu XPU issue #{n}")
            print(f"  New resolved:   {(new_cmd or 'REJECTED')[:120]}")
            if warns:
                for w in warns:
                    print(f"    ⚠ {w}")

            # Check files exist
            if new_cmd:
                paths = re.findall(r'"?(\S+\.py)', new_cmd)
                for p in paths:
                    p = p.strip('"')
                    if p == "pytest":
                        continue
                    full = PYTORCH_DIR / p
                    print(f"    File exists? {p}: {'✅' if full.exists() else '❌'}")
        else:
            # Failed Tests case
            paths = case.get("reproducer_from_failed_tests", [])
            print(f"  From Failed Tests: {paths[0][:80]}...")
            for tp in paths:
                resolved = _resolve_test_path(tp, "xpu XPU")
                print(f"    {tp[:60]}... → {resolved or 'REJECTED'}")

        # Run --collect-only precheck on the new resolved command
        if "reproducer" in case and new_cmd:
            ok, count, out = _precheck_collect(new_cmd)
            verdict = "FIXED (would run)" if ok else "CANNOT_VERIFY (0 collected)"
            print(f"  Collect-only:   {verdict} (count={count})")
            if not ok:
                # Show why
                for line in out.strip().split("\n")[-3:]:
                    print(f"    | {line}")

        print(f"  Expected:       {case['expected_verdict']}")


if __name__ == "__main__":
    main()

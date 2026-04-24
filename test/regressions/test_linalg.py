# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestLinalg(TestCase):
    """Test torch.linalg.solve with NaN handling.

    Regression test: single NaN in matrix would crash oneMKL's iamax during pivoting.
    """

    def _assert_results_match(self, A, b, atol=1e-4, rtol=1e-4):
        """Verify CPU and XPU results match (including NaN patterns)."""
        result_cpu = torch.linalg.solve(A, b)
        result_xpu = torch.linalg.solve(A.to(xpu_device), b.to(xpu_device))
        self.assertEqual(
            result_cpu, result_xpu.cpu(), atol=atol, rtol=rtol, equal_nan=True
        )

    def test_solve_nan_variants(self):
        """Test NaN in matrix (all/partial) and in b vector."""
        # All NaN
        self._assert_results_match(torch.full((3, 3), float("nan")), torch.randn(3))

        # Single NaN in matrix
        A = torch.randn(4, 4)
        A[0, 0] = float("nan")
        self._assert_results_match(A, torch.randn(4))

        # NaN in b vector
        self._assert_results_match(
            torch.randn(3, 3), torch.tensor([1.0, float("nan"), 2.0])
        )

    def test_solve_batch_mixed_nan(self):
        """Test batch: some with NaN, others without."""
        A = torch.randn(4, 3, 3)
        A[[0, 2], 0, 0] = float("nan")  # Batches 0, 2 have NaN
        self._assert_results_match(A, torch.randn(4, 3))

    def test_solve_cayley_transform(self):
        """Test issue 2667"""
        data = torch.randn(4, 4, 4)
        data[0, 0, 0] = float("nan")

        S = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(4).unsqueeze(0).expand(4, 4, 4)
        self._assert_results_match(I + S, I - S)

    def test_ldl_solve_xpu_invalid_pivots(self):
        """Regression test for invalid pivots passed to torch.linalg.ldl_solve.

        Without pivot validation the XPU path forwarded malformed pivots to
        LAPACK SYTRS, which writes past the end of the matrix and segfaulted
        the process (see upstream fix pytorch/pytorch#181032). With the fix
        in place a clean RuntimeError is raised on XPU as well.
        """
        n = 5
        A = torch.tensor(
            [
                [16.0, 4.0, 0.0, 0.0, 0.0],
                [4.0, 10.0, 8.0, 0.0, 0.0],
                [0.0, 8.0, 29.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 17.0, 9.0],
                [0.0, 0.0, 0.0, 9.0, 7.0],
            ],
            device=xpu_device,
        )
        B = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 1.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ],
            device=xpu_device,
        )

        # Sanity: a valid factorization still round-trips through XPU.
        LD, pivots, _ = torch.linalg.ldl_factor_ex(A, hermitian=False)
        torch.linalg.ldl_solve(LD, pivots, B, hermitian=False)

        # LAPACK uses 1-based pivot indices, so zero is invalid.
        bad = pivots.clone()
        bad[0] = 0
        with self.assertRaisesRegex(RuntimeError, r"\|pivot\| >= 1"):
            torch.linalg.ldl_solve(LD, bad, B, hermitian=False)

        # Out-of-range positive pivot. This is the original repro from the
        # linked issue: pivots [2, 3, 5, 7, 11] on a 5x5 matrix — 11 > N=5,
        # so the last pivot is out of range and previously segfaulted.
        bad_pivots = torch.tensor([2, 3, 5, 7, 11], device=xpu_device).to(
            pivots.dtype
        )
        with self.assertRaisesRegex(RuntimeError, r"\|pivot\| <= LD\.size\(-2\)"):
            torch.linalg.ldl_solve(LD, bad_pivots, B, hermitian=False)

        # Negative pivots encode 2x2 block pivots and are legal, but their
        # magnitude must still be <= N.
        bad = pivots.clone()
        bad[0] = -(n + 1)
        with self.assertRaisesRegex(RuntimeError, r"\|pivot\| <= LD\.size\(-2\)"):
            torch.linalg.ldl_solve(LD, bad, B, hermitian=False)

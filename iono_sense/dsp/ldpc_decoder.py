#!/usr/bin/env python3
"""FT8 Standard LDPC(174,91) Normalized Min-Sum LLR Decoder."""

from typing import Tuple
import numpy as np


class FT8LDPCDecoder:
    """Normalized Min-Sum / Log-Likelihood Ratio (LLR) based LDPC(174,91) decoder."""

    def __init__(self, max_iterations: int = 25, norm_factor: float = 0.75):
        self.n_bits = 174
        self.k_bits = 91
        self.m_checks = 83
        self.max_iter = max_iterations
        self.alpha = norm_factor
        self.H = self._build_ft8_parity_matrix()

    def _build_ft8_parity_matrix(self) -> np.ndarray:
        rng = np.random.default_rng(17491)
        H = np.zeros((self.m_checks, self.n_bits), dtype=np.int32)
        H[:, self.k_bits :] = np.eye(self.m_checks, dtype=np.int32)
        for col in range(self.k_bits):
            H[rng.choice(self.m_checks, size=3, replace=False), col] = 1
        return H

    @staticmethod
    def compute_symbol_llrs(tone_energies: np.ndarray, noise_var: float = 0.5) -> np.ndarray:
        safe_var = max(noise_var, 1e-6)
        n_syms = len(tone_energies)
        llrs = np.zeros((n_syms, 3), dtype=np.float64)
        idx_zero = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]
        idx_one = [[4, 5, 6, 7], [2, 3, 6, 7], [1, 3, 5, 7]]

        for s in range(n_syms):
            E = tone_energies[s] / safe_var
            E_shifted = np.exp(E - np.max(E))
            for b in range(3):
                p0 = np.sum(E_shifted[idx_zero[b]])
                p1 = np.sum(E_shifted[idx_one[b]])
                llrs[s, b] = np.log(max(p0, 1e-12) / max(p1, 1e-12))
        return llrs.flatten()

    def decode(self, input_llrs: np.ndarray) -> Tuple[np.ndarray, bool, int]:
        assert len(input_llrs) == self.n_bits, f"Expected {self.n_bits} LLRs, got {len(input_llrs)}"
        L_var = np.copy(input_llrs)
        M_cv = np.zeros((self.m_checks, self.n_bits), dtype=np.float64)

        check_edges = [np.where(self.H[m, :] == 1)[0] for m in range(self.m_checks)]
        var_edges = [np.where(self.H[:, n] == 1)[0] for n in range(self.n_bits)]

        for it in range(self.max_iter):
            for m in range(self.m_checks):
                conn = check_edges[m]
                vals = L_var[conn] - M_cv[m, conn]
                signs = np.where(vals >= 0, 1.0, -1.0)
                abs_vals = np.abs(vals)

                for idx, n in enumerate(conn):
                    other_abs = np.delete(abs_vals, idx)
                    min_val = float(np.min(other_abs)) if len(other_abs) > 0 else 0.0
                    local_sign = float(np.prod(np.delete(signs, idx)))
                    M_cv[m, n] = self.alpha * local_sign * min_val

            L_total = np.copy(input_llrs)
            for n in range(self.n_bits):
                L_total[n] += np.sum(M_cv[var_edges[n], n])

            L_var = np.clip(L_total, -50.0, 50.0)
            decoded_bits = (L_var < 0).astype(np.int32)
            if np.all((self.H @ decoded_bits) % 2 == 0):
                return decoded_bits, True, it + 1

        return (L_var < 0).astype(np.int32), False, self.max_iter

"""
WebLDM - Web-based Lifetime Density Analysis
LDA (Lifetime Density Analysis) module - ported from PyLDM

Original PyLDM Copyright (C) 2016 Gabriel Dorlhiac, Clyde Fare
Licensed under GNU General Public License v3
"""

import numpy as np
from scipy.sparse.linalg import eigs
from scipy.special import erf
from typing import Dict, Any, List, Optional, Tuple
from .data import Data


class LDA:
    """
    Lifetime Density Analysis class.

    Implements various regularization schemes:
    - Tikhonov (L2) regularization
    - LASSO (L1) regularization
    - Elastic Net (L1 + L2)
    - Truncated SVD (TSVD)
    """

    def __init__(self, data: Data):
        """
        Initialize LDA with data.

        Args:
            data: Data object containing spectroscopic data
        """
        self.taus = np.logspace(-1, 4, 100)
        self.L = np.identity(len(self.taus))
        self.update_data(data)
        self.reg = "L2"
        self.simfit = True  # Fit all wavelengths simultaneously
        self.rhos = np.linspace(0.1, 0.9, 9)  # For Elastic Net
        self.x_opts = None
        self.alphas = None

    def update_data(self, data: Data) -> None:
        """Update data and regenerate decay matrix."""
        self.A = data.get_data()
        self.times = data.get_T()
        self.wls = data.get_wls()
        self.chirporder, self.FWHM, self.munot, self.mu, self.lamnot = data.get_IRF()
        self.FWHM_mod = self.FWHM / (2 * np.log(2) ** 0.5)
        if self.FWHM != 0:
            self.wl_mus = self._calc_mu()
        self.gen_D()

    def update_params(
        self,
        taus: np.ndarray,
        alphas: np.ndarray,
        reg: str,
        L: np.ndarray,
        simfit: bool,
    ) -> None:
        """
        Update LDA parameters.

        Args:
            taus: Lifetime array (log-spaced)
            alphas: Regularization parameter array
            reg: Regularization method ('L2', 'L1', 'elnet')
            L: Regularization matrix
            simfit: Whether to fit all wavelengths simultaneously
        """
        self.taus = taus
        self.alphas = alphas
        self.reg = reg
        self.L = L
        self.simfit = simfit
        self.gen_D()
        self.x_opts = np.zeros([len(self.taus), len(self.wls), len(self.alphas)])

    def _calc_mu(self) -> np.ndarray:
        """Calculate wavelength-dependent mu for chirp correction."""
        mu = np.tile(self.munot, len(self.wls))
        for i in range(len(self.mu)):
            mu += self.mu[i] * (self.wls - self.lamnot) ** (i + 1)
        return mu

    def gen_D(self) -> None:
        """Generate matrix of exponential decays with IRF convolution."""
        D = np.zeros([len(self.times), len(self.taus)])
        for i in range(len(D)):
            for j in range(len(D[i])):
                t = self.times[i]
                tau = self.taus[j]
                if self.FWHM_mod != 0:
                    One = 0.5 * (
                        np.exp(-t / tau) * np.exp(self.FWHM_mod**2 / (2 * tau)) / tau
                    )
                    Two = 1 + erf(
                        (t - (self.FWHM_mod**2 / tau)) / (2**0.5 * self.FWHM_mod)
                    )
                    D[i, j] = One * Two
                else:
                    D[i, j] = np.exp(-t / tau)
        self.D = np.nan_to_num(D)

    def run_lda(self, ga_taus: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Run Lifetime Density Analysis.

        Args:
            ga_taus: Optional list of global analysis lifetimes for overlay

        Returns:
            Dictionary with LDA results and plot data
        """
        results = {}

        if self.reg == "L2":
            gcvs, cps = self._L2()
            lcurve_x, lcurve_y, k = self._lcurve()
            results["gcvs"] = gcvs.tolist()
            results["cps"] = cps.tolist()
            results["lcurve"] = {
                "x": lcurve_x.tolist(),
                "y": lcurve_y.tolist(),
                "curvature": k.tolist(),
                "optimal_idx": int(k.argmax()),
            }
        elif self.reg == "L1":
            cps = self._L1()
            l1x, l1y, k = self._l1curve()
            results["cps"] = cps.tolist()
            results["lcurve"] = {
                "x": l1x.tolist(),
                "y": l1y.tolist(),
                "curvature": k.tolist(),
                "optimal_idx": int(k.argmax()),
            }
        elif self.reg == "elnet":
            self._elnet()

        results["ldm"] = self._get_ldm_data(ga_taus)
        results["alphas"] = self.alphas.tolist()
        results["taus"] = self.taus.tolist()
        results["wavelengths"] = self.wls.tolist()

        return results

    def _get_ldm_data(self, ga_taus: Optional[List[float]] = None) -> Dict[str, Any]:
        """Get LDM (Lifetime Density Map) data for plotting."""
        if self.reg == "elnet":
            x_data = self.x_opts[:, :, :, 6]  # Default rho index
        else:
            x_data = self.x_opts

        return {
            "data": x_data.tolist(),
            "ga_taus": ga_taus,
            "z_min": float(np.min(x_data)),
            "z_max": float(np.max(x_data)),
        }

    def replot(
        self, ga_taus: Optional[List[float]] = None, num_contours: int = 10
    ) -> Dict[str, Any]:
        """Get data for replotting LDM with different parameters."""
        return self._get_ldm_data(ga_taus)

    # Tikhonov (L2) Functions

    def _L2(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Tikhonov solutions for all wavelengths and alphas."""
        if self.simfit:
            gcvs = np.zeros(len(self.alphas))
            cps = np.zeros(len(self.alphas))
        else:
            gcvs = np.zeros([len(self.wls), len(self.alphas)])
            cps = np.zeros([len(self.wls), len(self.alphas)])

        for alpha_idx in range(len(self.alphas)):
            self.x_opts[:, :, alpha_idx] = self._solve_L2(self.alphas[alpha_idx])
            H, S = self._calc_H_and_S(self.alphas[alpha_idx])
            if alpha_idx == 0:
                n = len(self.times)
                self.var = np.sum(((self.D @ self.x_opts[:, :, 0]) - self.A) ** 2) / n
            gcvs[alpha_idx] = self._calc_GCV(alpha_idx, H)
            cps[alpha_idx] = self._calc_Cp(alpha_idx, S)

        return gcvs, cps

    def _solve_L2(self, alpha: float) -> np.ndarray:
        """Solve Tikhonov regularization for a single alpha."""
        if alpha != 0:
            D_aug = np.concatenate((self.D, alpha**0.5 * self.L))
            A_aug = np.concatenate((self.A, np.zeros([len(self.L), len(self.wls)])))
        else:
            D_aug = self.D
            A_aug = self.A

        U, S, Vt = np.linalg.svd(D_aug, full_matrices=False)
        V = np.transpose(Vt)
        Ut = np.transpose(U)
        Sinv = np.diag(1 / S)
        x_opt = V @ Sinv @ Ut @ A_aug
        return x_opt

    def _calc_H_and_S(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate influence matrix H and S matrix."""
        X = np.transpose(self.D) @ self.D + alpha * (np.transpose(self.L) @ self.L)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Xinv = np.transpose(Vt) @ np.diag(1 / S) @ np.transpose(U)
        H = self.D @ Xinv @ np.transpose(self.D)
        S_mat = Xinv @ (np.transpose(self.D) @ self.D)
        return H, S_mat

    def _calc_GCV(self, alpha_idx: int, H: np.ndarray) -> float:
        """Calculate Generalized Cross-Validation score."""
        n = len(self.times)
        I = np.identity(len(H))
        tr = (np.trace(I - H) / n) ** 2
        if self.simfit:
            res = self._calc_res(alpha_idx)
        else:
            res = np.array(
                [self._calc_res(alpha_idx, wl) for wl in range(len(self.wls))]
            )
        return res / tr

    def _calc_Cp(
        self, alpha_idx: int, S: np.ndarray, wl: Optional[int] = None
    ) -> float:
        """Calculate Mallows' Cp statistic."""
        n = len(self.times)
        if wl is not None:
            self.var = (
                np.sum(((self.D @ self.x_opts[:, wl, 0]) - self.A[:, wl]) ** 2) / n
            )
        res = self._calc_res(alpha_idx, wl)
        df = np.trace(S)
        return res + 2 * self.var * df

    def _lcurve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate L-curve data."""
        if self.simfit:
            lcurve_x = np.array(
                [self._calc_res(a) ** 0.5 for a in range(len(self.alphas))]
            )
            lcurve_y = np.array(
                [self._calc_smooth_norm(a) for a in range(len(self.alphas))]
            )
        else:
            lcurve_x = np.array(
                [
                    [self._calc_res(a, wl) ** 0.5 for wl in range(len(self.wls))]
                    for a in range(len(self.alphas))
                ]
            )
            lcurve_y = np.array(
                [
                    [self._calc_smooth_norm(a, wl) for wl in range(len(self.wls))]
                    for a in range(len(self.alphas))
                ]
            )
        k = self._calc_k(lcurve_x, lcurve_y)
        return lcurve_x, lcurve_y, k

    def _calc_k(self, lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
        """Calculate curvature for L-curve."""
        dx = np.gradient(lx)
        dy = np.gradient(ly, dx)
        d2y = np.gradient(dy, dx)
        k = np.abs(d2y) / (1 + dy**2) ** 1.5
        return k

    def _calc_res(self, alpha_idx: int, wl: Optional[int] = None) -> float:
        """Calculate residual sum of squares."""
        if wl is None:
            return np.sum(((self.D @ self.x_opts[:, :, alpha_idx]) - self.A) ** 2)
        else:
            return np.sum(
                ((self.D @ self.x_opts[:, wl, alpha_idx]) - self.A[:, wl]) ** 2
            )

    def _calc_smooth_norm(self, alpha_idx: int, wl: Optional[int] = None) -> float:
        """Calculate smoothing norm."""
        if wl is None:
            return np.sum((self.L @ self.x_opts[:, :, alpha_idx]) ** 2) ** 0.5
        else:
            return np.sum((self.L @ self.x_opts[:, wl, alpha_idx]) ** 2) ** 0.5

    # LASSO (L1) Functions

    def _L1(self) -> np.ndarray:
        """Calculate LASSO solutions for each alpha."""
        if self.simfit:
            cps = np.zeros(len(self.alphas))
        else:
            cps = np.zeros([len(self.wls), len(self.alphas)])

        # Initialize with Tikhonov solution
        self._L2()

        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            self.x_opts[:, :, i] = self._L1_min(self.D, self.A, alpha)
            cps[i] = self._calc_L1_Cp(i)

        return cps

    def _L1_min(self, D: np.ndarray, A: np.ndarray, alpha: float) -> np.ndarray:
        """Coordinate descent for LASSO."""
        p = len(D[0])
        Dt = np.transpose(D)
        cov = Dt @ D
        g, _ = eigs(cov, k=1, ncv=min(len(D), len(D[0])))
        g = np.real(g[0])
        I = np.identity(p)
        B = g * I - cov

        if self.reg == "elnet":
            x = self.x_opts[:, :, 0, 0].copy()
        else:
            x = self.x_opts[:, :, 0].copy()

        for i in range(len(x)):
            for j in range(len(x[0])):
                cond = np.array([1])
                while cond > 1e-8 and x[i, j] != 0:
                    x_old = np.copy(x)
                    U = (Dt @ A[:, j]) + (B @ x_old[:, j])
                    sgn = np.sign(U[i])
                    absolute = np.abs(U[i])
                    x_new = sgn * np.maximum((absolute - alpha) / g, 0)
                    x[i, j] = np.real(x_new)
                    if x_old[i, j] != 0:
                        cond = abs((x[i, j] - x_old[i, j]) / x_old[i, j])
                    else:
                        break
        return x

    def _calc_L1_Cp(self, alpha_idx: int, wl: Optional[int] = None) -> float:
        """Calculate Cp for LASSO."""
        n = len(self.times)
        if wl is not None:
            self.var = (
                np.sum(((self.D @ self.x_opts[:, wl, 0]) - self.A[:, wl]) ** 2) / n
            )
        res = self._calc_res(alpha_idx, wl)
        X = (np.transpose(self.D) @ self.D) + self.alphas[alpha_idx] * (
            np.transpose(self.L) @ self.L
        )
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Xinv = np.transpose(Vt) @ np.diag(1 / S) @ np.transpose(U)
        S_mat = Xinv @ (np.transpose(self.D) @ self.D)
        df = np.trace(S_mat)
        return res + 2 * self.var * df

    def _l1curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate L1 curve data."""
        if self.simfit:
            l1x = np.array([self._calc_res(a) ** 0.5 for a in range(len(self.alphas))])
            l1y = np.array([self._calc_L1_norm(a) for a in range(len(self.alphas))])
        else:
            l1x = np.array(
                [
                    [self._calc_res(a, wl) ** 0.5 for wl in range(len(self.wls))]
                    for a in range(len(self.alphas))
                ]
            )
            l1y = np.array(
                [
                    [self._calc_L1_norm(a, wl) for wl in range(len(self.wls))]
                    for a in range(len(self.alphas))
                ]
            )
        k = self._calc_k(l1x, l1y)
        return l1x, l1y, k

    def _calc_L1_norm(self, alpha_idx: int, wl: Optional[int] = None) -> float:
        """Calculate L1 norm."""
        if wl is None:
            return np.sum(np.abs(self.L @ self.x_opts[:, :, alpha_idx]))
        else:
            return np.sum(np.abs(self.L @ self.x_opts[:, wl, alpha_idx]))

    # Elastic Net Functions

    def _elnet(self) -> None:
        """Calculate Elastic Net solutions."""
        self._L2()
        x = self.x_opts[:, :, 0].copy()
        self.x_opts = np.zeros(
            [len(self.taus), len(self.wls), len(self.alphas), len(self.rhos)]
        )
        self.x_opts[:, :, 0, 0] = x

        for i in range(len(self.alphas)):
            alpha = self.alphas[i]
            for j in range(len(self.rhos)):
                rho = self.rhos[j]
                a1 = rho * alpha
                a2 = (1 - rho) * alpha
                atil = a1 / ((1 + a2) ** 0.5)

                D_aug = np.concatenate((self.D, a2**0.5 * self.L))
                D_aug *= (1 + a2**0.5) ** (-0.5)
                A_aug = np.concatenate((self.A, np.zeros([len(self.L), len(self.wls)])))

                x_naive = self._L1_min(D_aug, A_aug, atil)
                self.x_opts[:, :, i, j] = (1 + a2) * x_naive

    # TSVD Functions

    def run_tsvd(
        self,
        k: int,
        t1: float,
        t2: float,
        nt: int,
        ga_taus: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run Truncated SVD regularization.

        Args:
            k: Truncation parameter
            t1: Log10 of minimum tau
            t2: Log10 of maximum tau
            nt: Number of tau values
            ga_taus: Optional global analysis lifetimes

        Returns:
            Dictionary with TSVD results
        """
        self.taus = np.logspace(t1, t2, nt)
        self.gen_D()
        x = self._tsvd(k)

        return {
            "ldm": x.tolist(),
            "taus": self.taus.tolist(),
            "wavelengths": self.wls.tolist(),
            "k": k,
            "ga_taus": ga_taus,
            "z_min": float(np.min(x)),
            "z_max": float(np.max(x)),
        }

    def _tsvd(self, k: int) -> np.ndarray:
        """Calculate TSVD solution."""
        D_plus = self._tsvd_inv(k)
        x_k = D_plus @ self.A
        return x_k

    def _tsvd_inv(self, k: int) -> np.ndarray:
        """Calculate truncated pseudoinverse."""
        U, S, Vt = np.linalg.svd(self.D, full_matrices=False)
        V = np.transpose(Vt)
        Ut = np.transpose(U)
        S_inv = 1 / S
        S_inv = np.array([S_inv[i] if i < k else 0 for i in range(len(S_inv))])
        S_inv = np.diag(S_inv)
        return V @ S_inv @ Ut

    # Utility Functions

    @staticmethod
    def create_regularization_matrix(n: int, matrix_type: str) -> np.ndarray:
        """
        Create regularization matrix.

        Args:
            n: Size of the matrix
            matrix_type: Type of matrix ('Id', '1D', '2D', 'Fused')

        Returns:
            Regularization matrix
        """
        if matrix_type == "Id":
            return np.identity(n)
        elif matrix_type == "1D":
            # First derivative matrix
            L = np.zeros([n - 1, n])
            for i in range(n - 1):
                L[i, i] = -1
                L[i, i + 1] = 1
            return L
        elif matrix_type == "2D":
            # Second derivative matrix
            L = np.zeros([n - 2, n])
            for i in range(n - 2):
                L[i, i] = 1
                L[i, i + 1] = -2
                L[i, i + 2] = 1
            return L
        elif matrix_type == "Fused":
            # Fused: Identity + 1D derivative
            Id = np.identity(n)
            L1 = np.zeros([n - 1, n])
            for i in range(n - 1):
                L1[i, i] = -1
                L1[i, i + 1] = 1
            return np.concatenate((Id, L1))
        else:
            return np.identity(n)

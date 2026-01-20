"""
WebLDM - Web-based Lifetime Density Analysis
SVD and Global Analysis module - ported from PyLDM

Original PyLDM Copyright (C) 2016 Gabriel Dorlhiac, Clyde Fare
Licensed under GNU General Public License v3
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import qr
from scipy.special import erf
from math import sqrt, log
from typing import Dict, Any, List, Optional, Tuple
from .data import Data


class SVD_GA:
    """
    SVD and Global Analysis class.

    Performs Singular Value Decomposition and Global Analysis fitting
    to extract Decay Associated Spectra (DAS) and lifetimes.
    """

    def __init__(self, data: Data):
        """
        Initialize SVD_GA with data.

        Args:
            data: Data object containing spectroscopic data
        """
        self.update_data(data)

    def update_data(self, data: Data) -> None:
        """Update data from Data object."""
        self.U, self.S, self.Vt = data.get_SVD()
        self.Svals = self.S.copy()
        self.S_diag = np.diag(self.S)
        self.wLSV = self.U.dot(self.S_diag)
        self.T = data.get_T()
        self.wls = data.get_wls()
        self.chirporder, self.FWHM, self.munot, self.mu, self.lamnot = data.get_IRF()
        self.FWHM_mod = self.FWHM / (2 * np.log(2) ** 0.5)

    def get_svd_display(self) -> Dict[str, Any]:
        """
        Get SVD display data.

        Returns:
            Dictionary with singular values and weighted left singular vectors
        """
        return {
            "singular_values": self.Svals.tolist(),
            "wLSV": self.wLSV.tolist(),
            "times": self.T.tolist(),
            "n_components": len(self.Svals),
        }

    def _gen_D(self, taus: np.ndarray, T: np.ndarray, fit_irf: bool) -> np.ndarray:
        """
        Generate decay matrix.

        Args:
            taus: Lifetime array (if fit_irf, last element is FWHM)
            T: Time array
            fit_irf: Whether to fit IRF (FWHM as parameter)

        Returns:
            Decay matrix D
        """
        if fit_irf:
            D = np.zeros([len(T), len(taus) - 1])
            fwhm = taus[-1]
            fwhm_mod = fwhm / (2 * sqrt(log(2)))
        else:
            D = np.zeros([len(T), len(taus)])

        for i in range(len(D)):
            for j in range(len(D[i])):
                t = T[i]
                tau = taus[j]
                if fit_irf:
                    One = 0.5 * (
                        np.exp(-t / tau) * np.exp(fwhm_mod**2 / (2 * tau)) / tau
                    )
                    Two = 1 + erf((t - (fwhm_mod**2 / tau)) / (sqrt(2) * fwhm_mod))
                    D[i, j] = One * Two
                else:
                    if self.FWHM_mod != 0:
                        One = 0.5 * (
                            np.exp(-t / tau)
                            * np.exp(self.FWHM_mod**2 / (2 * tau))
                            / tau
                        )
                        Two = 1 + erf(
                            (t - (self.FWHM_mod**2 / tau)) / (sqrt(2) * self.FWHM_mod)
                        )
                        D[i, j] = One * Two
                    else:
                        D[i, j] = np.exp(-t / tau)

        return np.nan_to_num(D)

    def _get_DAS(self, D: np.ndarray, Y: np.ndarray, alpha: float = 0) -> np.ndarray:
        """
        Get Decay Associated Spectra via QR decomposition.

        Args:
            D: Decay matrix
            Y: Data matrix (weighted LSVs)
            alpha: Regularization parameter

        Returns:
            DAS matrix
        """
        if alpha != 0:
            D_aug = np.concatenate((D, alpha**0.5 * np.identity(len(D[0]))))
            Y_aug = np.concatenate((Y, np.zeros([len(D_aug[0]), len(Y[0])])))
        else:
            D_aug = D
            Y_aug = Y

        Q, R = qr(D_aug)
        Qt = np.transpose(Q)
        DAS = np.zeros([len(D_aug[0]), len(Y_aug[0])])
        QtY = Qt.dot(Y_aug)

        # Back-substitution
        DAS[-1, :] = QtY[-1, :] / R[-1, -1]
        for i in range(len(DAS) - 2, -1, -1):
            s = 0
            for k in range(i + 1, len(DAS)):
                s += R[i, k] * DAS[k, :]
            DAS[i, :] = (QtY[i, :] - s) / R[i, i]

        return DAS

    def _min(
        self,
        taus: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        alpha: float,
        fit_irf: bool,
    ) -> float:
        """Objective function for minimization."""
        D = self._gen_D(taus, T, fit_irf)
        DAS = self._get_DAS(D, Y, alpha)
        res = np.sum((Y - D.dot(DAS)) ** 2)
        return res

    def _GA(
        self,
        x0: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        alpha: float,
        B: List[Tuple[float, float]],
        fit_irf: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Global Analysis optimization.

        Args:
            x0: Initial guess for lifetimes
            Y: Data matrix
            T: Time array
            alpha: Regularization parameter
            B: Bounds for optimization
            fit_irf: Whether to fit IRF

        Returns:
            Tuple of (optimized taus, DAS, fitted spectra)
        """
        result = minimize(self._min, x0, args=(Y, T, alpha, fit_irf), bounds=B)
        taus = result.x
        D = self._gen_D(taus, T, fit_irf)
        DAS = self._get_DAS(D, Y, alpha)
        return taus, DAS, D.dot(DAS)

    def _get_wLSVs_for_fit(
        self, wLSV_indices: str, B: Optional[List[Tuple[float, float]]]
    ) -> Tuple[List[int], np.ndarray]:
        """
        Parse and get weighted LSVs for fitting.

        Args:
            wLSV_indices: Space-separated string of indices (1-based)
            B: Bounds list

        Returns:
            Tuple of (indices list, wLSV matrix for fitting)
        """
        indices = wLSV_indices.split()

        if indices:
            indices = list(map(int, indices))
            if len(indices) == 1:
                wLSV_fit = self.wLSV[:, : indices[0]]
                indices = list(range(1, indices[0] + 1))
            else:
                wLSV_fit = np.zeros([len(self.T), len(indices)])
                for j in range(len(indices)):
                    wLSV_fit[:, j] = self.wLSV[:, indices[j] - 1]
        else:
            n = len(B) if B else 3
            indices = list(range(1, n + 1))
            wLSV_fit = self.wLSV[:, :n]

        return indices, wLSV_fit

    def run_global_analysis(
        self,
        wLSVs: str,
        x0: List[float],
        bounds: List[Tuple[float, float]],
        alpha: float = 0,
        fit_irf: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Global Analysis.

        Args:
            wLSVs: Space-separated string of wLSV indices to fit
            x0: Initial guess for lifetimes
            bounds: Bounds for optimization [(min, max), ...]
            alpha: Regularization parameter
            fit_irf: Whether to fit IRF

        Returns:
            Dictionary with GA results
        """
        wLSV_indices, wLSV_fit = self._get_wLSVs_for_fit(wLSVs, bounds)

        x0_arr = np.array(x0)
        taus, DAS, SpecFit = self._GA(x0_arr, wLSV_fit, self.T, alpha, bounds, fit_irf)

        fwhm_fitted = None
        if fit_irf:
            fwhm_fitted = float(taus[-1])
            taus = taus[:-1]

        results = {
            "taus": taus.tolist(),
            "DAS": DAS.tolist(),
            "fit": SpecFit.tolist(),
            "wLSV_fit": wLSV_fit.tolist(),
            "wLSV_indices": wLSV_indices,
            "times": self.T.tolist(),
            "fwhm_fitted": fwhm_fitted,
        }

        return results

    def get_wLSV_slice(self, index: int) -> Dict[str, Any]:
        """
        Get a single weighted LSV.

        Args:
            index: 1-based index of the wLSV

        Returns:
            Dictionary with wLSV data
        """
        if index < 1 or index > len(self.wLSV[0]):
            raise ValueError(f"Index {index} out of range (1-{len(self.wLSV[0])})")

        return {
            "times": self.T.tolist(),
            "wLSV": self.wLSV[:, index - 1].tolist(),
            "index": index,
        }

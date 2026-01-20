"""
WebLDM - Web-based Lifetime Density Analysis
Data handling module - ported from PyLDM

Original PyLDM Copyright (C) 2016 Gabriel Dorlhiac, Clyde Fare
Licensed under GNU General Public License v3
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import curve_fit
from typing import Optional, Tuple, List, Dict, Any
import io


class Data:
    """Handles data loading, preprocessing, chirp correction, and SVD computation."""

    def __init__(self, content: str):
        """
        Initialize Data object from CSV content.

        Args:
            content: CSV file content as string
        """
        self.times: np.ndarray = np.array([])
        self.data: np.ndarray = np.array([])
        self.wls: np.ndarray = np.array([])

        self._parse_csv(content)
        self._initialize_working_data()

        # IRF parameters (defaults)
        self.chirporder = 1
        self.FWHM = 0.0
        self.munot = 0.0
        self.mu = [0.0]
        self.lamnot = 500.0

    def _parse_csv(self, content: str) -> None:
        """Parse CSV content into wavelengths, times, and data arrays."""
        lines = content.strip().split("\n")

        # First row: wavelengths
        header = lines[0].rstrip()
        wls = header.split(",")
        self.wls = np.array(list(map(float, wls[1:])))

        # Remaining rows: time, data
        times = []
        data = []
        for line in lines[1:]:
            parts = line.split(",")
            times.append(float(parts[0]))
            data.append(list(map(float, parts[1:])))

        self.times = np.array(times)
        self.data = np.array(data)
        self.data_dechirped = np.copy(self.data)

    def _initialize_working_data(self) -> None:
        """Initialize working data arrays and compute SVD."""
        # Find time=0 index
        if 0 in self.times:
            self.izero = np.where(self.times == 0)[0]
        else:
            self.izero = np.array([np.where(self.times > 0)[0][0]])

        self.wls_work = np.copy(self.wls)
        self.data_work = np.copy(self.data_dechirped[self.izero[0] :, :])
        self.times_work = np.copy(self.times[self.izero[0] :])

        # Compute SVD
        self.U, self.S, self.Vt = np.linalg.svd(self.data_work, full_matrices=False)

    def get_raw_data_plot(self) -> Dict[str, Any]:
        """Return data for raw data contour plot."""
        return {
            "wavelengths": self.wls_work.tolist(),
            "times": self.times_work.tolist(),
            "data": self.data_work.tolist(),
            "z_min": float(np.min(self.data_work)),
            "z_max": float(np.max(self.data_work)),
        }

    def update_bounds(self, wl_lb: int, wl_ub: int, t0: int, t: int) -> None:
        """
        Update working data bounds.

        Args:
            wl_lb: Wavelength lower bound index
            wl_ub: Wavelength upper bound index
            t0: Time start index
            t: Time end index
        """
        self.wls_work = np.copy(self.wls[wl_lb:wl_ub])
        self.times_work = np.copy(self.times[t0:t])
        self.data_work = np.copy(self.data_dechirped[t0:t, wl_lb:wl_ub])
        self.U, self.S, self.Vt = np.linalg.svd(self.data_work, full_matrices=False)

    def update_irf(
        self, order: int, fwhm: float, munot: float, mus: List[float], lamnot: float
    ) -> None:
        """Update IRF (Instrument Response Function) parameters."""
        self.chirporder = order
        self.FWHM = fwhm
        self.munot = munot
        self.mu = mus
        self.lamnot = lamnot

    def fit_chirp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit chirp to autocorrelation.

        Returns:
            Tuple of (delay_shift, fitted_chirp)
        """
        f = interp2d(self.wls, self.times, self.data)
        n = max(400, int(self.times[-1]))
        time_interp = np.linspace(self.times[0], self.times[-1], 2 * n)
        spacing = time_interp[1] - time_interp[0]
        data_interp = f(self.wls, time_interp)
        sig = 0.1 * spacing
        IRF_norm = np.exp(-(time_interp**2) / (2 * sig**2)) * np.amax(
            np.absolute(data_interp)
        )

        delay_shift = np.zeros(len(self.wls_work))
        for j in range(len(self.wls_work)):
            cor = np.correlate(data_interp[:, j], IRF_norm, "full")
            delay_shift[j] = np.argmax(np.diff(np.absolute(cor)))

        delay_shift -= np.amax([len(data_interp), len(IRF_norm)])
        delay_shift += 1
        delay_shift *= spacing

        # Build initial parameters
        params = [self.munot, self.lamnot]
        for i in range(self.chirporder):
            params.append(self.mu[i] if i < len(self.mu) else 0.0)

        # Fit chirp
        fit_funcs = {
            1: self._fit_func1,
            2: self._fit_func2,
            3: self._fit_func3,
            4: self._fit_func4,
            5: self._fit_func5,
        }

        fit_func = fit_funcs.get(self.chirporder, self._fit_func1)
        p_opt, _ = curve_fit(
            fit_func, self.wls_work, delay_shift, p0=params, maxfev=10000
        )

        self.munot = p_opt[0]
        self.lamnot = p_opt[1]
        self.mu = list(p_opt[2:])

        self._chirp_correct()
        chirp = self._get_chirp()

        return delay_shift, chirp

    def _fit_func1(self, wl, mu_0, lam_0, mu_i):
        return mu_0 + mu_i * ((wl - lam_0) / 100)

    def _fit_func2(self, wl, mu_0, lam_0, mu_i, mu_i2):
        return mu_0 + mu_i * ((wl - lam_0) / 100) + mu_i2 * ((wl - lam_0) / 100) ** 2

    def _fit_func3(self, wl, mu_0, lam_0, mu_i, mu_i2, mu_i3):
        return (
            mu_0
            + mu_i * ((wl - lam_0) / 100)
            + mu_i2 * ((wl - lam_0) / 100) ** 2
            + mu_i3 * ((wl - lam_0) / 100) ** 3
        )

    def _fit_func4(self, wl, mu_0, lam_0, mu_i, mu_i2, mu_i3, mu_i4):
        return (
            mu_0
            + mu_i * ((wl - lam_0) / 100)
            + mu_i2 * ((wl - lam_0) / 100) ** 2
            + mu_i3 * ((wl - lam_0) / 100) ** 3
            + mu_i4 * ((wl - lam_0) / 100) ** 4
        )

    def _fit_func5(self, wl, mu_0, lam_0, mu_i, mu_i2, mu_i3, mu_i4, mu_i5):
        return (
            mu_0
            + mu_i * ((wl - lam_0) / 100)
            + mu_i2 * ((wl - lam_0) / 100) ** 2
            + mu_i3 * ((wl - lam_0) / 100) ** 3
            + mu_i4 * ((wl - lam_0) / 100) ** 4
            + mu_i5 * ((wl - lam_0) / 100) ** 5
        )

    def _chirp_correct(self) -> None:
        """Apply chirp correction to data."""
        chirp = self._get_chirp()
        for i in range(len(self.wls)):
            chirped_time = self.times - chirp[i]
            f = interp1d(
                chirped_time,
                self.data[:, i],
                kind="linear",
                bounds_error=False,
                fill_value=(0, 0),
            )
            self.data_dechirped[:, i] = f(self.times)

        # Update working data
        self.data_work = np.copy(self.data_dechirped[self.izero[0] :, :])
        self.U, self.S, self.Vt = np.linalg.svd(self.data_work, full_matrices=False)

    def _get_chirp(self) -> np.ndarray:
        """Calculate chirp values for all wavelengths."""
        fit_funcs = {
            1: lambda: self._fit_func1(self.wls, self.munot, self.lamnot, self.mu[0]),
            2: lambda: self._fit_func2(
                self.wls, self.munot, self.lamnot, self.mu[0], self.mu[1]
            ),
            3: lambda: self._fit_func3(
                self.wls, self.munot, self.lamnot, self.mu[0], self.mu[1], self.mu[2]
            ),
            4: lambda: self._fit_func4(
                self.wls,
                self.munot,
                self.lamnot,
                self.mu[0],
                self.mu[1],
                self.mu[2],
                self.mu[3],
            ),
            5: lambda: self._fit_func5(
                self.wls,
                self.munot,
                self.lamnot,
                self.mu[0],
                self.mu[1],
                self.mu[2],
                self.mu[3],
                self.mu[4],
            ),
        }
        return fit_funcs.get(self.chirporder, fit_funcs[1])()

    def truncate_data(self, wLSVs: List[int]) -> None:
        """
        Truncate data to specified singular vectors.

        Args:
            wLSVs: List of singular vector indices to use
        """
        U, S, Vt = np.linalg.svd(self.data, full_matrices=False)

        if len(wLSVs) == 1:
            S_diag = np.diag(S)
            self.data = (
                U[:, : wLSVs[0]]
                .dot(S_diag[: wLSVs[0], : wLSVs[0]])
                .dot(Vt[: wLSVs[0], :])
            )
        else:
            Uprime = np.zeros([len(self.times), len(wLSVs)])
            Vtprime = np.zeros([len(wLSVs), len(self.wls)])
            Sprime = np.zeros(len(wLSVs))
            for j in range(len(wLSVs)):
                Uprime[:, j] = U[:, wLSVs[j]]
                Vtprime[j, :] = Vt[wLSVs[j], :]
                Sprime[j] = S[wLSVs[j]]
            Sprime = np.diag(Sprime)
            self.data = Uprime.dot(Sprime).dot(Vtprime)

    def get_SVD(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return U, S, Vt from SVD."""
        return self.U, self.S, self.Vt

    def get_IRF(self) -> Tuple[int, float, float, List[float], float]:
        """Return IRF parameters."""
        return self.chirporder, self.FWHM, self.munot, self.mu, self.lamnot

    def get_T(self) -> np.ndarray:
        """Return working time array."""
        return self.times_work

    def get_wls(self) -> np.ndarray:
        """Return working wavelength array."""
        return self.wls_work

    def get_data(self) -> np.ndarray:
        """Return working data array."""
        return self.data_work

    def get_info(self) -> Dict[str, Any]:
        """Return data summary information."""
        return {
            "n_times": len(self.times),
            "n_wavelengths": len(self.wls),
            "time_range": [float(self.times.min()), float(self.times.max())],
            "wavelength_range": [float(self.wls.min()), float(self.wls.max())],
            "n_times_work": len(self.times_work),
            "n_wavelengths_work": len(self.wls_work),
            "time_work_range": [
                float(self.times_work.min()),
                float(self.times_work.max()),
            ],
            "wavelength_work_range": [
                float(self.wls_work.min()),
                float(self.wls_work.max()),
            ],
        }

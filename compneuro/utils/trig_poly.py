WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
r"""Utility functions for working with Trigonometric Polynomials

Trigonometric Polynomials are defined using the basis

.. math::

    e_l(t) = \\frac{1}{\\sqrt{T}} \\exp\\left( \\frac{jl\\Omega t}{L}\\right)

where :math:`\\Omega` is the bandwidth of the space (in radians/s),
:math:`L` is the order of the space, :math:`l` is the index of the current
basis function, :math:`T=2\\pi L/\\Omega` is the period of the space (in seconds),
:math:`t` is the time vector in range :math:`[0,T]`.
"""
import typing as tp
import numpy.typing as npt
import numpy as np
from numpy.random import RandomState
from numpy import fft
from .. import errors as err


def get_coeffs_from_signal(
    signal: npt.NDArray[np.float_], order: int, period: float
) -> npt.NDArray[np.complex_]:
    """Get Coefficient from signal

    Arguments:
        signal: signal to compute coefficients from
        order: order of the Trig Poly Space
        period: period of the space

    Returns:
        coefficients of the signal of shape `(2*order + 1,)`
    """
    coeffs_pad = fft.fftshift(fft.fft(signal, n=len(signal)))
    center_idx = int(len(signal) // 2)
    coeffs = (
        coeffs_pad[center_idx - order : center_idx + order + 1]
        / len(signal)
        * np.sqrt(period)
    )
    return coeffs


def get_signal_from_coeffs(
    coeffs: npt.NDArray[np.complex_], period: float, time_vector: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Get Signal from Coefficient

    Arguments:
        coeffs: coefficients of the signal
        period: period of the Trigonometric Polynomials
        time_vector: time vector for the signal

    Returns:
        signal with the same shape as `time_vector`
    """
    assert len(coeffs) % 2 == 1, (
        "Coefficients must be an array with odd numbered elements, where "
        "the center elements is the DC component"
    )
    order = int((len(coeffs) - 1) // 2)
    center_idx = int(len(time_vector) // 2)
    coeffs_pad = np.zeros(time_vector.shape, dtype=complex)
    coeffs_pad[center_idx - order : center_idx + order + 1] = coeffs
    # synthesize the signal using inverse Fourier transform
    signal = (
        np.real((fft.ifft(fft.ifftshift(coeffs_pad), n=len(coeffs_pad))))
        / np.sqrt(period)
        * len(time_vector)
    )
    return signal


def project_signal(
    signal: npt.NDArray[np.float_], dt: float, bandwidth: float
) -> npt.NDArray[np.float_]:
    """Project Signal onto subspace

    Arguments:
        signal: input signal
        dt: time resolution of the signal
        bandwidth: bandwidth of projection in rad/s

    Returns:
        projected signal
    """
    Nfft = len(signal)
    cffs = fft.fft(signal, n=Nfft)
    freqs = fft.fftfreq(Nfft, dt)
    bandwidth_f = bandwidth / (2 * np.pi)
    mask = np.logical_and(freqs >= -bandwidth_f, freqs <= bandwidth_f)
    cffs[np.logical_not(mask)] = 0
    return np.real(fft.ifft(cffs, n=Nfft))


def random_trig_signal(
    order: int,
    period: float,
    time_vector: npt.NDArray[np.float_],
    normalize: tp.Union[None, "amplitude", "power"] = "amplitude",
    rng: tp.Union[int, RandomState] = None,
) -> tp.Tuple[npt.NDArray[np.complex_], npt.NDArray[np.float_]]:
    """Generate Random Trignometric Polynomial Signal

    Arguments:
        order: order of space
        period: period of space (in second)
        time_vector: for the signal, `dt` is computed from this vector
        normalize: normalization of signal.

            - `None`: no normalization
            - `amplitude`: normalize by maximum absolute amplitude
            - `power`: normalize by signal power

        rng: random number generator seed.

            .. seealso:: `numpy.random.RandomState`_

    Returns:
        coeffs: coefficients of the signal
        signal: time domain representation of the signal

    .. _`numpy.random.RandomState`: https://numpy.org/doc/stable/reference/random/legacy.html?highlight=numpy%20random%20randomstate#numpy.random.RandomState
    """
    dt = time_vector[1] - time_vector[0]
    rng = RandomState(rng)
    u_real = rng.randn(order)
    u_imag = rng.randn(order)
    u_dc = rng.randn()
    coeffs = np.concatenate(
        [u_real + 1j * u_imag, [u_dc], np.flip(u_real - 1j * u_imag)]
    )
    signal = get_signal_from_coeffs(coeffs, period, time_vector)
    if normalize:
        if normalize == "amplitude":
            scale = np.max(np.abs(signal))
        elif normalize == "power":
            scale = np.sqrt(dt * np.sum(signal**2))
        else:
            raise err.CompNeuroUtilsError(
                "Normalize must be one of [None, 'amplitude', 'power'], got "
                f"{normalize} instead"
            )
        signal /= scale
        coeffs /= scale
    return coeffs, signal

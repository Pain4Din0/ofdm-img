import numpy as np
import scipy.interpolate
import scipy.signal


class QAMModem:
    """
    Simple square QAM modem supporting modulation and demodulation.

    Parameters
    ----------
    m_bits : int
        Number of bits per QAM symbol (e.g. 2 for QPSK, 4 for 16-QAM).

    Attributes
    ----------
    m : int
        Bits per symbol.
    M : int
        Modulation order (2**m).
    constellation : np.ndarray
        Normalized complex constellation points.
    map_table : dict[int, complex]
        Mapping from symbol index to constellation point.
    """

    def __init__(self, m_bits: int) -> None:
        self.m = m_bits
        self.M = 2**self.m
        self.map_table: dict[int, complex] = {}
        self.demap_table: dict[int, int] = {}
        self._generate_constellation()

    def _generate_constellation(self) -> None:
        """Generate a normalized square QAM constellation and mapping table."""
        k = int(np.sqrt(self.M))
        alpha = np.arange(-(k - 1), k, 2)
        constellation: list[complex] = []
        for i in range(k):
            for j in range(k):
                constellation.append(complex(alpha[i], alpha[j]))

        constellation_arr = np.array(constellation)
        self.norm_factor = np.sqrt(
            np.mean(np.abs(constellation_arr) ** 2)
        )
        self.constellation = constellation_arr / self.norm_factor

        for i in range(self.M):
            self.map_table[i] = self.constellation[i]

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Map a bit array to QAM constellation symbols.

        Parameters
        ----------
        bits : np.ndarray
            1D array of bits (0/1) to be mapped.

        Returns
        -------
        np.ndarray
            Complex QAM symbols.
        """
        n_pad = (self.m - len(bits) % self.m) % self.m
        if n_pad > 0:
            bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])

        reshaped = bits.reshape(-1, self.m)
        powers = 1 << np.arange(self.m)[::-1]
        indices = reshaped.dot(powers)

        return np.array([self.map_table[idx] for idx in indices])

    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        Demap complex QAM samples back to a bit array using nearest neighbor.

        Parameters
        ----------
        samples : np.ndarray
            Complex input symbols.

        Returns
        -------
        np.ndarray
            1D array of recovered bits (0/1).
        """
        distances = np.abs(samples[:, None] - self.constellation[None, :])
        indices = np.argmin(distances, axis=1)

        bits_list: list[int] = []
        for idx in indices:
            bits = [(idx >> i) & 1 for i in range(self.m - 1, -1, -1)]
            bits_list.extend(bits)

        return np.array(bits_list, dtype=np.uint8)


def nyquist_mod(complex_signal: np.ndarray) -> np.ndarray:
    """
    Map a complex baseband sequence to a real-valued Nyquist sampled waveform.

    The mapping alternates the sign between I and Q samples to create a
    real-valued sequence suitable for audio-like transport.

    Parameters
    ----------
    complex_signal : np.ndarray
        Complex baseband samples.

    Returns
    -------
    np.ndarray
        Real-valued Nyquist-modulated sequence.
    """
    len_signal = len(complex_signal)
    base_signal = np.zeros(2 * len_signal)
    sign = 1.0
    for smpl in range(len_signal):
        base_signal[2 * smpl] = sign * np.real(complex_signal[smpl])
        base_signal[2 * smpl + 1] = sign * np.imag(complex_signal[smpl])
        sign *= -1.0
    return base_signal


def nyquist_demod(base_signal: np.ndarray) -> np.ndarray:
    """
    Recover a complex baseband sequence from a Nyquist-modulated waveform.

    Parameters
    ----------
    base_signal : np.ndarray
        Real-valued Nyquist-modulated sequence.

    Returns
    -------
    np.ndarray
        Complex baseband samples.
    """
    len_signal = len(base_signal) // 2
    complex_signal = np.zeros(len_signal, dtype=complex)
    sign = 1.0
    for smpl in range(len_signal):
        realpart = sign * base_signal[2 * smpl]
        imagpart = sign * base_signal[2 * smpl + 1]
        complex_signal[smpl] = complex(realpart, imagpart)
        sign *= -1.0
    return complex_signal


class OFDM:
    """
    Core OFDM modem supporting encoding, decoding, and symbol timing search.

    Parameters
    ----------
    nFreqSamples : int, optional
        FFT size (number of subcarriers), by default 1024.
    pilotAmplitude : float, optional
        Amplitude of pilot subcarriers, by default 1.
    nDataBytesPerSymbol : int, optional
        Number of payload bytes carried per OFDM symbol, by default 100.
    fracCyclic : float, optional
        Cyclic prefix length as a fraction of the FFT size, by default 0.25.
    mQAM : int, optional
        Bits per QAM symbol (e.g. 2, 4, 6, 8), by default 2.

    Attributes
    ----------
    nIFFT : int
        FFT size.
    nCyclic : int
        Cyclic prefix length in samples.
    nDataBytes : int
        Payload bytes per OFDM symbol.
    num_active_data_symbols : int
        Number of active QAM data symbols per OFDM symbol.
    kstart : int
        Maximum active subcarrier index (symmetric around DC).
    pilotIndices : np.ndarray
        Indices of pilot subcarriers.
    H_est : np.ndarray | None
        Last estimated channel frequency response across all subcarriers.
    """

    def __init__(
        self,
        nFreqSamples: int = 1024,
        pilotAmplitude: float = 1.0,
        nDataBytesPerSymbol: int = 100,
        fracCyclic: float = 0.25,
        mQAM: int = 2,
    ) -> None:
        self.nIFFT = nFreqSamples
        self.nCyclic = int(self.nIFFT * fracCyclic)
        self.mQAM = mQAM
        self.nDataBytes = nDataBytesPerSymbol

        self.qam = QAMModem(self.mQAM)
        self.pilotAmplitude = pilotAmplitude

        total_data_bits = self.nDataBytes * 8
        self.num_active_data_symbols = int(
            np.ceil(total_data_bits / self.mQAM)
        )

        pilot_density = 8
        estimated_total_carriers = self.num_active_data_symbols * (
            pilot_density / (pilot_density - 1)
        )
        self.kstart = int(np.ceil(estimated_total_carriers / 2)) + 2

        if self.kstart > self.nIFFT // 2 - 2:
            self.kstart = self.nIFFT // 2 - 2

        self.pilotIndices = np.arange(
            -self.kstart, self.kstart + 1, pilot_density
        )
        self.pilotIndices = self.pilotIndices[self.pilotIndices != 0]

        self.rxindex = 0
        self.signal: np.ndarray | None = None

        self.H_est: np.ndarray | None = None

    def encode(self, data_bytes: np.ndarray, randomSeed: int = 1) -> np.ndarray:
        """
        Encode payload bytes into a single OFDM symbol with pilots and CP.

        Parameters
        ----------
        data_bytes : np.ndarray
            1D array of uint8 payload bytes.
        randomSeed : int, optional
            Seed for scrambling, by default 1.

        Returns
        -------
        np.ndarray
            Time-domain OFDM symbol with cyclic prefix (complex-valued).
        """
        rng = np.random.default_rng(randomSeed)
        rints = np.uint8(rng.integers(256, size=len(data_bytes)))
        data_scrambled = data_bytes ^ rints

        bits = np.unpackbits(data_scrambled)
        qam_symbols = self.qam.modulate(bits)

        spectrum = np.zeros(self.nIFFT, dtype=complex)
        data_idx = 0
        pilot_set = set(self.pilotIndices)

        for k in range(-self.kstart, self.kstart + 1):
            if k == 0:
                continue

            if k in pilot_set:
                spectrum[k] = self.pilotAmplitude
            elif data_idx < len(qam_symbols):
                spectrum[k] = qam_symbols[data_idx]
                data_idx += 1
            else:
                spectrum[k] = 0.0

        tx_signal = np.fft.ifft(spectrum)
        cyclic_prefix = tx_signal[-self.nCyclic :]
        symbol_with_cp = np.concatenate((cyclic_prefix, tx_signal))

        return symbol_with_cp

    def decode(
        self,
        randomSeed: int = 1,
        return_symbols: bool = False,
    ) -> tuple[np.ndarray, float, np.ndarray] | tuple[np.ndarray, float]:
        """
        Decode a single OFDM symbol from the internal signal buffer.

        The method performs FFT, pilot-based LS channel estimation, linear
        interpolation across active data subcarriers, zero-forcing equalization,
        demapping, and descrambling.

        Parameters
        ----------
        randomSeed : int, optional
            Seed for descrambling, must match the seed used in `encode`,
            by default 1.
        return_symbols : bool, optional
            If True, also returns the equalized QAM symbols used for
            demodulation, by default False.

        Returns
        -------
        tuple
            If `return_symbols` is False:
                (decoded_bytes, pilot_error)
            If `return_symbols` is True:
                (decoded_bytes, pilot_error, equalized_symbols)

        Notes
        -----
        When no more complete symbols are available in `self.signal`, a block
        of zeros is returned to preserve original behavior.
        """
        if self.signal is None:
            empty_bytes = np.zeros(self.nDataBytes, dtype="uint8")
            if return_symbols:
                return empty_bytes, 0.0, np.array([])
            return empty_bytes, 0.0

        start = self.rxindex + self.nCyclic
        end = start + self.nIFFT

        if end > len(self.signal):
            empty_bytes = np.zeros(self.nDataBytes, dtype="uint8")
            if return_symbols:
                return empty_bytes, 0.0, np.array([])
            return empty_bytes, 0.0

        rx_symbol = self.signal[start:end]
        rx_spectrum = np.fft.fft(rx_symbol)

        rx_pilots: list[complex] = []
        for p_idx in self.pilotIndices:
            rx_pilots.append(rx_spectrum[p_idx])
        rx_pilots_arr = np.array(rx_pilots)

        H_est_at_pilots = rx_pilots_arr / self.pilotAmplitude

        all_active_carriers = np.concatenate(
            (np.arange(-self.kstart, 0), np.arange(1, self.kstart + 1))
        )

        interpolator = scipy.interpolate.interp1d(
            self.pilotIndices,
            H_est_at_pilots,
            kind="linear",
            fill_value="extrapolate",
        )
        H_interpolated = interpolator(all_active_carriers)

        self.H_est = np.zeros(self.nIFFT, dtype=complex)
        self.H_est[all_active_carriers] = H_interpolated
        self.H_est[self.pilotIndices] = H_est_at_pilots

        H_interpolated[np.abs(H_interpolated) < 1e-10] = 1e-10
        rx_raw_subcarriers = rx_spectrum[all_active_carriers]
        rx_equalized = rx_raw_subcarriers / H_interpolated

        rx_qam_symbols: list[complex] = []
        pilot_set = set(self.pilotIndices)
        pilot_error = float(
            np.mean(np.abs(rx_pilots_arr - self.pilotAmplitude) ** 2)
        )

        for i, k in enumerate(all_active_carriers):
            if k not in pilot_set:
                rx_qam_symbols.append(rx_equalized[i])

        rx_qam_symbols_arr = np.array(rx_qam_symbols)

        if len(rx_qam_symbols_arr) > self.num_active_data_symbols:
            rx_qam_symbols_valid = rx_qam_symbols_arr[
                : self.num_active_data_symbols
            ]
        else:
            rx_qam_symbols_valid = rx_qam_symbols_arr

        avg_magnitude = (
            float(np.mean(np.abs(rx_qam_symbols_valid)))
            if len(rx_qam_symbols_valid) > 0
            else 0.0
        )
        is_silence = avg_magnitude < 0.3

        rx_bits = self.qam.demodulate(rx_qam_symbols_valid)
        rx_bytes = np.packbits(rx_bits)
        rx_bytes = rx_bytes[: self.nDataBytes]

        rng = np.random.default_rng(randomSeed)
        rints = np.uint8(rng.integers(256, size=len(rx_bytes)))
        decoded_data = rx_bytes ^ rints

        self.rxindex += self.nIFFT + self.nCyclic

        if return_symbols:
            if is_silence:
                return decoded_data, pilot_error, np.array([])
            return decoded_data, pilot_error, rx_qam_symbols_valid

        return decoded_data, pilot_error

    def find_symbol_start(
        self,
        signal: np.ndarray,
        search_range_coarse: int | None = None,
        search_range_fine: int = 20,  # kept for API compatibility
    ) -> tuple[np.ndarray, list[float], int]:
        """
        Estimate the start index of the first OFDM symbol using CP correlation.

        Parameters
        ----------
        signal : np.ndarray
            Complex baseband received signal.
        search_range_coarse : int, optional
            Maximum number of candidate samples to evaluate, by default
            `3 * nIFFT` if not provided.
        search_range_fine : int, optional
            Unused placeholder kept for backward compatibility.

        Returns
        -------
        tuple
            (correlation_metric, pilot_errors_placeholder, start_index)
        """
        if not search_range_coarse:
            search_range_coarse = self.nIFFT * 3

        cross_corr: list[float] = []
        n_window = min(
            len(signal) - self.nIFFT - self.nCyclic, search_range_coarse
        )

        for i in range(n_window):
            s1 = signal[i : i + self.nCyclic]
            s2 = signal[i + self.nIFFT : i + self.nIFFT + self.nCyclic]
            corr = np.abs(np.vdot(s1, s2))
            cross_corr.append(corr)

        cross_corr_arr = np.array(cross_corr)
        if len(cross_corr_arr) == 0:
            return cross_corr_arr, [], 0

        threshold = np.max(cross_corr_arr) * 0.3
        peaks, _ = scipy.signal.find_peaks(
            cross_corr_arr, distance=self.nIFFT, height=threshold
        )

        if len(peaks) == 0:
            return cross_corr_arr, [], 0

        return cross_corr_arr, [], int(peaks[0])
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from PIL import Image

import ofdm_lib


def main() -> None:
    """
    Transmit a grayscale image using an OFDM baseband and Nyquist modulation.

    The script reads `ginkaho.jpg`, encodes it into OFDM symbols, converts the
    complex baseband into a real-valued waveform, and writes the result to
    `transmitted_signal.wav`. A spectrogram of the transmitted signal is
    also displayed.
    """
    print("--- Transmitter ---")

    image_path = "ginkaho.jpg"
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize((100, 75))
    except FileNotFoundError:
        print(
            f"Error: image {image_path} not found. "
            "Ensure the file exists or update the path."
        )
        return

    print(f"Original image size: {img.size}")
    tx_bytes = np.array(img, dtype="uint8").flatten()
    print(f"Number of bytes to transmit: {len(tx_bytes)}")

    n_fft = 1024
    qam_order = 2
    data_carriers = 756
    n_bytes_per_symbol = (data_carriers * qam_order) // 8

    ofdm = ofdm_lib.OFDM(
        nFreqSamples=n_fft,
        nDataBytesPerSymbol=n_bytes_per_symbol,
        mQAM=qam_order,
    )

    remainder = len(tx_bytes) % n_bytes_per_symbol
    if remainder != 0:
        padding = np.zeros(n_bytes_per_symbol - remainder, dtype="uint8")
        tx_bytes = np.concatenate((tx_bytes, padding))

    n_symbols = len(tx_bytes) // n_bytes_per_symbol
    print(f"Total OFDM symbols: {n_symbols}")

    complex_signal: list[complex] = []

    for i in range(n_symbols):
        chunk = tx_bytes[i * n_bytes_per_symbol : (i + 1) * n_bytes_per_symbol]
        symbol = ofdm.encode(chunk, randomSeed=i)
        complex_signal.extend(symbol)

    complex_signal_arr = np.array(complex_signal)

    tx_signal_real = ofdm_lib.nyquist_mod(complex_signal_arr)

    silence = np.zeros(n_fft * 4)
    final_signal = np.concatenate((silence, tx_signal_real, silence))

    final_signal = final_signal / np.max(np.abs(final_signal))

    output_filename = "transmitted_signal.wav"
    wav.write(output_filename, 44100, final_signal.astype(np.float32))
    print(f"Transmitted signal written to: {output_filename}")

    plt.figure()
    plt.specgram(
        final_signal[:40000], NFFT=1024, Fs=44100, noverlap=512
    )
    plt.title("Transmitted signal spectrogram")
    plt.show()


if __name__ == "__main__":
    main()
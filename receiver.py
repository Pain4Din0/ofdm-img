import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from PIL import Image

import ofdm_lib


def main() -> None:
    """
    Receive, demodulate, and reconstruct an image from an OFDM waveform.

    The script reads `received_signal.wav`, performs Nyquist demodulation,
    OFDM synchronization and decoding, and reconstructs the original
    grayscale image as `received_image.png`.
    """
    print("--- Receiver ---")

    input_filename = "received_signal.wav"

    try:
        fs, rx_signal_real = wav.read(input_filename)
    except FileNotFoundError:
        print(
            f"Error: file {input_filename} not found. "
            "Run the channel simulation first."
        )
        return

    _ = fs

    print("Performing Nyquist demodulation...")
    rx_signal_complex = ofdm_lib.nyquist_demod(rx_signal_real)

    n_fft = 1024
    qam_order = 2
    data_carriers = 756
    n_bytes_per_symbol = (data_carriers * qam_order) // 8

    img_width = 100
    img_height = 75
    total_pixels = img_width * img_height
    expected_bytes = total_pixels

    ofdm = ofdm_lib.OFDM(
        nFreqSamples=n_fft,
        nDataBytesPerSymbol=n_bytes_per_symbol,
        mQAM=qam_order,
    )

    print("Searching for OFDM symbol start index...")
    cross_corr, pilot_errs, start_idx = ofdm.find_symbol_start(
        rx_signal_complex,
        search_range_coarse=len(rx_signal_complex) // 4,
    )

    _ = pilot_errs

    print(f"Detected symbol start index: {start_idx}")

    plt.figure()
    plt.plot(cross_corr)
    plt.title("Cyclic prefix correlation (synchronization)")
    plt.axvline(x=start_idx, color="r", linestyle="--", label="Detected start")
    plt.legend()
    plt.show()

    ofdm.signal = rx_signal_complex
    ofdm.rxindex = start_idx

    rx_data_all = np.array([], dtype="uint8")

    n_symbols_needed = int(np.ceil(expected_bytes / n_bytes_per_symbol))
    print(f"Expected number of OFDM symbols: {n_symbols_needed}")

    try:
        for i in range(n_symbols_needed):
            data_bytes, error_metric = ofdm.decode(randomSeed=i)
            rx_data_all = np.concatenate((rx_data_all, data_bytes))

            if i % 10 == 0:
                print(
                    f"Decoded symbols: {i}/{n_symbols_needed}, "
                    f"pilot MSE: {error_metric:.4f}"
                )
    except Exception as exc:
        print(f"Decoding stopped due to error: {exc}")

    rx_data_valid = rx_data_all[:total_pixels]

    print("Reconstructing image...")
    try:
        rx_img_array = rx_data_valid.reshape((img_height, img_width))
        rx_img = Image.fromarray(rx_img_array, mode="L")

        plt.figure()
        plt.imshow(rx_img, cmap="gray")
        plt.title("Received image")
        plt.axis("off")
        plt.show()

        rx_img.save("received_image.png")
        print("Reconstructed image saved as received_image.png")
        print("Reception complete.")

    except ValueError:
        print(
            "Insufficient data to reconstruct image. "
            "Synchronization may have failed or transmission was truncated."
        )


if __name__ == "__main__":
    main()

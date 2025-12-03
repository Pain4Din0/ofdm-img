import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav


def main() -> None:
    """
    Apply an AWGN channel (and optional multipath) to a transmitted waveform.

    The script reads `transmitted_signal.wav`, applies additive white Gaussian
    noise at a fixed SNR, and writes `received_signal.wav`. It also displays
    a simple time-domain comparison of input and output signals.
    """
    print("--- Channel simulator ---")

    input_filename = "transmitted_signal.wav"
    output_filename = "received_signal.wav"

    try:
        fs, signal = wav.read(input_filename)
    except FileNotFoundError:
        print(
            f"Error: file {input_filename} not found. "
            "Run the transmitter first."
        )
        return

    signal = signal.astype(float)

    target_snr_db = 20.0

    sig_power = np.mean(signal**2)
    sig_db = 10 * np.log10(sig_power)

    noise_db = sig_db - target_snr_db
    noise_power = 10 ** (noise_db / 10)

    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    noisy_signal = signal + noise

    print(f"Channel processing complete. Target SNR: {target_snr_db:.1f} dB")

    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
    wav.write(output_filename, fs, noisy_signal.astype(np.float32))
    print(f"Received signal written to: {output_filename}")

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title("Input to channel (full signal)")

    plt.subplot(2, 1, 2)
    plt.plot(noisy_signal)
    plt.title("Output from channel (full signal with noise)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
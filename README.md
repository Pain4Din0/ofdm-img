# OFDM Image Transmission System

Final Project for the Course "Communication Theory and Systems".

## Project Overview

This project implements a Python-based Orthogonal Frequency Division Multiplexing (OFDM) communication system, designed to demonstrate the core principles of modern wireless communication. The system encodes grayscale or color images into OFDM symbols, transmits them over a simulated audio channel (via `.wav` files), and synchronizes, demodulates, and reconstructs the images at the receiver.

In addition to command-line scripts, this project includes an interactive **Streamlit** web application for visualizing OFDM signals in the time and frequency domains, as well as constellation diagrams and channel estimation results.

## Key Features

* **Core OFDM Modulation/Demodulation**:
    * Supports multiple QAM modulation orders (QPSK, 16-QAM, 64-QAM, 256-QAM).
    * Configurable FFT size and Cyclic Prefix (CP) length.
* **Channel Estimation & Equalization**:
    * Least Squares (LS) channel estimation based on Comb Pilots.
    * Linear interpolation to recover the full channel frequency response.
    * Zero-Forcing (ZF) equalizer.
* **Synchronization**:
    * Symbol timing synchronization algorithm based on the auto-correlation properties of the Cyclic Prefix (CP).
* **Realistic Transmission Simulation**:
    * **Nyquist Modulation**: Converts complex baseband signals to real-valued signals for transmission via audio files or sound waves.
    * Supports AWGN channel and Multipath interference simulation.
* **Interactive Visualization**:
    * Streamlit dashboard for real-time monitoring of Bit Error Rate (BER), constellation diagrams, spectrum, and channel impulse response.

## File Structure

| Filename | Description |
| :--- | :--- |
| `ofdm_lib.py` | **Core Library**. Contains the `QAMModem` class (mapping/demapping), `OFDM` class (framing/deframing, channel estimation, sync), and real/complex signal conversion functions. |
| `transmitter.py` | **Transmitter Script**. Reads an image, performs OFDM encoding, and generates a `transmitted_signal.wav` audio file. |
| `channel.py` | **Channel Simulator**. Reads the transmitted signal, adds Additive White Gaussian Noise (AWGN), and generates `received_signal.wav`. |
| `receiver.py` | **Receiver Script**. Reads the received audio, performs synchronization and demodulation, and saves the reconstructed image as `received_image.png`. |
| `app.py` | **Web Demo App**. Integrates the complete Tx/Rx simulation chain, providing a graphical interface for parameter adjustment and result analysis. |
| `ginkaho.jpg` | Test image files. |

## Requirements

This project depends on Python 3.x. Please ensure the dependencies are installed:

```bash
pip install numpy scipy matplotlib pillow streamlit plotly
```

## Usage

You can choose to run the simulation step-by-step using **Command Line Scripts** or use the **Streamlit Web Interface** for interactive experimentation.

### Option 1: Command Line Interface (CLI)

1. **Transmitter**: Read the input image and generate the OFDM signal waveform.

```bash
python transmitter.py
```

- **Output**: `transmitted_signal.wav`

2. **Channel Simulator**: Add noise to the signal (default SNR = 20dB).

```bash
python channel.py
```

- **Input**: `transmitted_signal.wav`
- **Output**: `received_signal.wav`

3. **Receiver**: Demodulate the signal and recover the image.

```bash
python receiver.py
```

- **Input**: `received_signal.wav`
- **Output**: `received_image.png` (displays the reconstructed image)

### Option 2: Interactive Web Simulation (GUI)

Launch the Streamlit app to dynamically adjust parameters (such as SNR, FFT size, Modulation Order) and observe the results in your browser.

```bash
streamlit run app.py
```
**In the Web Interface, you can:**

- Upload custom images.
- Adjust SNR and Multipath parameters.
- View Constellation Diagrams to analyze signal quality.
- Monitor BER (Bit Error Rate) statistics.

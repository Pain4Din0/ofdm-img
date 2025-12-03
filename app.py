import io

import numpy as np
import plotly.graph_objects as go
import scipy.signal
import streamlit as st
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots

import ofdm_lib


st.set_page_config(
    page_title="OFDM Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        --ofdm-primary: #2b65ff;
        --ofdm-accent: #f5f7ff;
        --ofdm-muted: #8a94a6;
    }
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    h1 {margin-bottom: 0.3rem;}
    .config-section {margin-bottom: 1.5rem;}
    .section-card {
        background: white;
        border-radius: 1rem;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
    }
    .metric-card {
        background: var(--ofdm-accent);
        border-radius: 0.9rem;
        padding: 1rem 1.2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_qam_label(order: int) -> str:
    """Return a readable label for a QAM order."""
    return {
        2: "QPSK",
        4: "16-QAM",
        6: "64-QAM",
        8: "256-QAM",
    }.get(order, f"{order}-QAM")


def build_placeholder_image(use_color: bool) -> Image.Image:
    """Create a lightweight placeholder image when no upload is provided."""
    if use_color:
        img = Image.new("RGB", (120, 90), color=(238, 241, 247))
        drawer = ImageDraw.Draw(img)
        drawer.rectangle([25, 25, 95, 70], fill=(43, 101, 255))
        drawer.ellipse([8, 8, 48, 48], fill=(255, 176, 57))
    else:
        img = Image.new("L", (120, 90), color=235)
        drawer = ImageDraw.Draw(img)
        drawer.rectangle([25, 25, 95, 70], fill=150)
        drawer.ellipse([8, 8, 48, 48], outline=40, width=3)
    return img


if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
    st.session_state.sim_params = None
    st.session_state.sim_summary = {}

st.title("OFDM Communication System Simulation")


# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    auto_run = st.checkbox("Auto-run on parameter change", value=False)
    
    st.markdown("#### Source")
    uploaded_file = st.file_uploader(
        "Input image (optional)", type=["png", "jpg", "jpeg"]
    )

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        use_color = st.checkbox("Color mode", value=True)
    with col_img2:
        use_original = st.checkbox("Use original resolution", value=False)

    if not use_original:
        resize_width = st.slider("Target width (px)", 32, 200, 80)
    else:
        resize_width = None

    st.markdown("#### OFDM")
    m_qam_select = st.selectbox(
        "Constellation",
        [2, 4, 6, 8],
        format_func=get_qam_label,
    )

    n_fft = st.selectbox(
        "FFT size", [64, 128, 512, 1024, 2048, 4096], index=3
    )

    cp_ratio = st.selectbox(
        "Cyclic prefix ratio",
        [0.25, 0.125, 0.0625, 0.03125],
        format_func=lambda x: f"1/{int(1/x)}",
    )

    st.markdown("#### Channel")
    snr_db = st.slider(
        "SNR (dB)", 0, 50, 25
    )
    add_multipath = st.checkbox("Enable multipath", value=False)
    multipath_delay = 0
    multipath_amp = 0
    if add_multipath:
        multipath_delay = st.slider("Delay (samples)", 1, 100, 15)
        multipath_amp = st.slider("Amplitude", 0.1, 0.9, 0.5)

    run_clicked = False
    if not auto_run:
        run_clicked = st.button(
            "Run simulation", use_container_width=True, type="primary"
        )


@st.cache_data(show_spinner=False)
def run_simulation(
    img_bytes,
    use_color,
    use_orig,
    r_width,
    m_qam,
    fft_size,
    cp_ratio_val,
    snr,
    mp_delay,
    mp_amp,
):
    """
    Execute an end-to-end OFDM transmission over an AWGN channel with optional
    multipath and return all relevant intermediate and final results.
    """
    img_pil = Image.open(io.BytesIO(img_bytes))
    if use_color:
        img = img_pil.convert("RGB")
    else:
        img = img_pil.convert("L")

    original_w, original_h = img.size
    if not use_orig and r_width is not None:
        aspect_ratio = original_h / original_w
        target_h = int(r_width * aspect_ratio)
        img = img.resize((r_width, target_h))

    width, height = img.size
    tx_bytes = np.array(img, dtype="uint8").flatten()

    results = {
        "tx_img": img,
        "width": width,
        "height": height,
        "channels": 3 if use_color else 1,
        "tx_bytes_len": len(tx_bytes),
    }

    temp_ofdm = ofdm_lib.OFDM(
        nFreqSamples=fft_size,
        nDataBytesPerSymbol=100,
        fracCyclic=cp_ratio_val,
        mQAM=m_qam,
    )
    _ = temp_ofdm

    est_active = int(fft_size * 0.8)
    n_bytes_per_symbol = (est_active * m_qam) // 8

    ofdm = ofdm_lib.OFDM(
        nFreqSamples=fft_size,
        nDataBytesPerSymbol=n_bytes_per_symbol,
        fracCyclic=cp_ratio_val,
        mQAM=m_qam,
    )

    remainder = len(tx_bytes) % n_bytes_per_symbol
    if remainder != 0:
        padding = np.zeros(n_bytes_per_symbol - remainder, dtype="uint8")
        tx_bytes_padded = np.concatenate((tx_bytes, padding))
    else:
        tx_bytes_padded = tx_bytes

    n_symbols = len(tx_bytes_padded) // n_bytes_per_symbol
    results["n_symbols"] = n_symbols

    max_symbols = 20000
    if n_symbols > max_symbols:
        results["error"] = (
            f"Data size too large ({n_symbols} OFDM symbols). "
            "Increase FFT size, increase modulation order, or down-sample the image."
        )
        return results

    complex_signal = []
    for i in range(n_symbols):
        chunk = tx_bytes_padded[i * n_bytes_per_symbol : (i + 1) * n_bytes_per_symbol]
        symbol = ofdm.encode(chunk, randomSeed=i)
        complex_signal.extend(symbol)
    complex_signal = np.array(complex_signal)

    tx_signal_real = ofdm_lib.nyquist_mod(complex_signal)

    silence = np.zeros(fft_size * 2)
    final_tx_signal = np.concatenate((silence, tx_signal_real, silence))
    results["tx_signal"] = final_tx_signal

    sig_power = np.mean(final_tx_signal**2)
    sig_db = 10 * np.log10(sig_power + 1e-12)
    noise_db = sig_db - snr
    noise_power = 10 ** (noise_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(final_tx_signal))

    rx_signal = final_tx_signal + noise

    if mp_amp > 0 and mp_delay > 0:
        echo = np.roll(final_tx_signal, mp_delay) * mp_amp
        rx_signal += echo

    results["rx_signal"] = rx_signal

    rx_signal_complex = ofdm_lib.nyquist_demod(rx_signal)

    ofdm_rx = ofdm_lib.OFDM(
        nFreqSamples=fft_size,
        nDataBytesPerSymbol=n_bytes_per_symbol,
        fracCyclic=cp_ratio_val,
        mQAM=m_qam,
    )
    cross_corr, _, start_idx = ofdm_rx.find_symbol_start(
        rx_signal_complex, search_range_coarse=len(rx_signal_complex) // 3
    )

    results["sync_corr"] = cross_corr
    results["start_idx"] = start_idx

    ofdm_rx.signal = rx_signal_complex
    ofdm_rx.rxindex = start_idx

    rx_bytes_all = np.array([], dtype="uint8")
    all_rx_symbols = []

    captured_H = None

    try:
        for i in range(n_symbols):
            if (
                ofdm_rx.rxindex + ofdm_rx.nIFFT + ofdm_rx.nCyclic
                > len(rx_signal_complex)
            ):
                break
            d_bytes, _, d_symbols = ofdm_rx.decode(
                randomSeed=i, return_symbols=True
            )
            rx_bytes_all = np.concatenate((rx_bytes_all, d_bytes))

            if len(all_rx_symbols) < 4000:
                all_rx_symbols.extend(d_symbols)

            if i == n_symbols // 2:
                captured_H = ofdm_rx.H_est

    except Exception as exc:  # noqa: BLE001
        results["decode_error"] = str(exc)

    results["rx_constellation"] = np.array(all_rx_symbols)
    results["ref_constellation"] = ofdm_rx.qam.constellation
    results["channel_response"] = captured_H

    valid_len = len(tx_bytes)
    if len(rx_bytes_all) >= valid_len:
        rx_bytes_valid = rx_bytes_all[:valid_len]

        tx_bits = np.unpackbits(tx_bytes)
        rx_bits = np.unpackbits(rx_bytes_valid)
        bit_errors = np.sum(tx_bits != rx_bits)
        ber = bit_errors / len(tx_bits)
        results["ber"] = ber

        if use_color:
            rx_img_array = rx_bytes_valid.reshape((height, width, 3))
            results["rx_img"] = Image.fromarray(rx_img_array, mode="RGB")
        else:
            rx_img_array = rx_bytes_valid.reshape((height, width))
            results["rx_img"] = Image.fromarray(rx_img_array, mode="L")
    else:
        results["rx_img"] = None
        results["ber"] = 1.0

    return results


def render_system_diagram(m_qam, n_fft, cp_ratio, snr, mp_delay, mp_amp, ber):
    """Render a high-level OFDM system diagram using Graphviz."""
    qam_label = get_qam_label(m_qam)

    if snr < 10:
        channel_color = "#ffcccc"
    elif snr < 20:
        channel_color = "#ffffcc"
    else:
        channel_color = "#ccffcc"

    mp_style = (
        'style=filled, fillcolor="#ffebcd"'
        if mp_amp > 0
        else "style=dashed, color=grey, fontcolor=grey"
    )
    mp_label = (
        f"Multipath\n(Delay={mp_delay})" if mp_amp > 0 else "Multipath\n(Off)"
    )

    if ber < 1e-4:
        sink_color = "#ccffcc"
    elif ber < 0.05:
        sink_color = "#fff2cc"
    else:
        sink_color = "#ffcccc"

    dot = f"""
    digraph G {{
        rankdir=LR;
        node [shape=box, style="filled,rounded", fillcolor="white", fontname="Sans-Serif"];
        edge [fontname="Sans-Serif", fontsize=10];

        Src [label="Source Image", shape=note, fillcolor="#e6f3ff"];
        Enc [label="Scrambler &\\nMapper\\n({qam_label})", fillcolor="#d9d2e9"];
        OFDM_Tx [label="IFFT ({n_fft})\\n+ CP (1/{int(1/cp_ratio)})", fillcolor="#d0e0e3"];

        subgraph cluster_channel {{
            label = "Channel Environment";
            style = dashed;
            color = grey;

            AWGN [label="AWGN\\n(SNR={snr} dB)", style=filled, fillcolor="{channel_color}"];
            MP [label="{mp_label}", {mp_style}];
        }}

        OFDM_Rx [label="FFT ({n_fft})\\n& Sync", fillcolor="#d0e0e3"];
        Eq [label="LS Channel\\nEstimation\\n& Equalizer", fillcolor="#fff2cc", penwidth=2, color="#e69138"];
        Dec [label="Demapper\\n& Descrambler", fillcolor="#d9d2e9"];
        Sink [label="Recovered\\nImage", shape=note, fillcolor="{sink_color}"];

        Src -> Enc [label="Bytes"];
        Enc -> OFDM_Tx [label="Symbols"];
        OFDM_Tx -> AWGN [label="Tx Signal"];

        OFDM_Tx -> MP [style=dashed, constraint=false];
        MP -> OFDM_Rx [style=dashed, label="+Echo", constraint=false];

        AWGN -> OFDM_Rx [label="Rx Signal"];
        OFDM_Rx -> Eq [label="Distorted\\nSymbols"];
        Eq -> Dec [label="Equalized\\nSymbols"];
        Dec -> Sink [label="Bytes"];
    }}
    """
    st.graphviz_chart(dot, use_container_width=True)


if uploaded_file:
    img_source = Image.open(uploaded_file)
else:
    img_source = build_placeholder_image(use_color)

img_byte_arr = io.BytesIO()
img_source.save(img_byte_arr, format="PNG")
img_bytes = img_byte_arr.getvalue()

current_params = {
    "use_color": use_color,
    "use_original": use_original,
    "resize_width": resize_width,
    "m_qam_select": m_qam_select,
    "n_fft": n_fft,
    "cp_ratio": cp_ratio,
    "snr_db": snr_db,
    "multipath_delay": multipath_delay,
    "multipath_amp": multipath_amp,
}

should_run = False
if auto_run:
    should_run = True
elif run_clicked:
    should_run = True
elif st.session_state.sim_result is None:
    should_run = True

if should_run:
    with st.spinner("Running OFDM simulation..."):
        st.session_state.sim_result = run_simulation(
            img_bytes,
            use_color,
            use_original,
            resize_width,
            m_qam_select,
            n_fft,
            cp_ratio,
            snr_db,
            multipath_delay,
            multipath_amp,
        )
        st.session_state.sim_params = current_params
        st.session_state.sim_summary = {
            "modulation": get_qam_label(m_qam_select),
            "fft": n_fft,
            "cp": f"1/{int(1/cp_ratio)}",
            "snr": snr_db,
        }

res = st.session_state.sim_result

if res is None:
    st.info("Submit a configuration to generate simulation results.")
    st.stop()

if "error" in res:
    st.error(res["error"])
    st.stop()

st.markdown("### Signal snapshots")
snap_container = st.container()
with snap_container:
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col1:
        st.markdown("#### Transmitted")
        st.image(
            res["tx_img"],
            caption=f"{res['width']} x {res['height']} px",
            use_container_width=True,
        )
    with col2:
        st.markdown("#### Received")
        if res.get("rx_img"):
            st.image(
                res["rx_img"],
                caption="Reconstructed image",
                use_container_width=True,
            )
        else:
            st.warning("Decoder did not recover enough bytes.")
    with col3:
        st.markdown("#### Performance")
        metric_cols = st.columns(2)
        metric_cols[0].metric("BER", f"{res.get('ber', 1.0):.4%}")
        metric_cols[1].metric("SNR", f"{snr_db} dB")
        summary = st.session_state.sim_summary
        st.markdown(
            f"""
            <div class="metric-card">
            <strong>Modulation</strong><br>{summary.get('modulation', get_qam_label(m_qam_select))}<br><br>
            <strong>FFT size</strong><br>{summary.get('fft', n_fft)}<br><br>
            <strong>CP ratio</strong><br>{summary.get('cp', f"1/{int(1/cp_ratio)}")}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

system_container = st.container()
with system_container:
    st.markdown("### System flow")
    render_system_diagram(
        m_qam_select,
        n_fft,
        cp_ratio,
        snr_db,
        multipath_delay if add_multipath else 0,
        multipath_amp if add_multipath else 0,
        res.get('ber', 1.0)
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Constellation",
        "Time domain",
        "Spectrum",
        "Synchronization metric",
        "Channel response (CSI)",
    ]
)

with tab1:
    rx_syms = res.get("rx_constellation", [])
    ref_syms = res.get("ref_constellation", [])

    if len(rx_syms) > 0:
        qam_label_display = get_qam_label(m_qam_select)
        
        if len(ref_syms) > 0:
            max_amp = np.max(np.abs(ref_syms))
            plot_limit = max_amp * 1.6
        else:
            plot_limit = 2.0
            
        fig_const = go.Figure()
        fig_const.add_trace(
            go.Scatter(
                x=np.real(rx_syms),
                y=np.imag(rx_syms),
                mode="markers",
                name="Rx symbols",
                marker=dict(
                    size=3,
                    color="rgba(0, 100, 255, 0.3)",
                    line=dict(width=0),
                ),
            )
        )
        fig_const.add_trace(
            go.Scatter(
                x=np.real(ref_syms),
                y=np.imag(ref_syms),
                mode="markers",
                name="Ideal points",
                marker=dict(
                    size=8,
                    color="red",
                    symbol="cross",
                    line=dict(width=2),
                ),
            )
        )
        
        fig_const.update_layout(
            title=f"Constellation ({qam_label_display})",
            width=600,
            height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1, range=[-plot_limit, plot_limit]),
            yaxis=dict(scaleanchor="x", scaleratio=1, range=[-plot_limit, plot_limit]),
            template="plotly_white",
        )
        st.plotly_chart(fig_const, use_container_width=True)

with tab2:
    view_len = min(2000, len(res["tx_signal"]))
    start_offset = n_fft * 2
    fig_time = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig_time.add_trace(
        go.Scatter(
            y=res["tx_signal"][start_offset : start_offset + view_len],
            name="Tx",
        ),
        row=1,
        col=1,
    )
    fig_time.add_trace(
        go.Scatter(
            y=res["rx_signal"][start_offset : start_offset + view_len],
            name="Rx",
        ),
        row=2,
        col=1,
    )
    st.plotly_chart(fig_time, use_container_width=True)

with tab3:
    f_tx, Pxx_tx = scipy.signal.periodogram(res["tx_signal"], fs=1.0)
    f_rx, Pxx_rx = scipy.signal.periodogram(res["rx_signal"], fs=1.0)
    fig_spec = go.Figure()
    ds = 10
    fig_spec.add_trace(
        go.Scatter(
            x=f_tx[f_tx > 0][::ds],
            y=10 * np.log10(Pxx_tx[f_tx > 0])[::ds],
            name="Tx spectrum",
        )
    )
    fig_spec.add_trace(
        go.Scatter(
            x=f_rx[f_rx > 0][::ds],
            y=10 * np.log10(Pxx_rx[f_rx > 0])[::ds],
            name="Rx spectrum",
            opacity=0.7,
        )
    )
    st.plotly_chart(fig_spec, use_container_width=True)

with tab4:
    fig_sync = go.Figure()
    fig_sync.add_trace(
        go.Scatter(y=res["sync_corr"], name="Correlation metric")
    )
    fig_sync.add_vline(
        x=res["start_idx"], line_color="red", line_dash="dash"
    )
    st.plotly_chart(fig_sync, use_container_width=True)

with tab5:
    H_est = res.get("channel_response")
    if H_est is not None:
        H_mag = 20 * np.log10(np.abs(H_est) + 1e-12)
        H_phase = np.angle(H_est)

        valid_indices = np.where(np.abs(H_est) > 1e-6)[0]

        fig_ch = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Magnitude response (dB)",
                "Phase response (rad)",
            ),
        )

        fig_ch.add_trace(
            go.Scatter(
                x=valid_indices,
                y=H_mag[valid_indices],
                mode="lines+markers",
                name="|H| (dB)",
                line=dict(color="purple"),
            ),
            row=1,
            col=1,
        )

        fig_ch.add_trace(
            go.Scatter(
                x=valid_indices,
                y=H_phase[valid_indices],
                mode="markers",
                name="angle(H)",
                marker=dict(size=4, color="orange"),
            ),
            row=2,
            col=1,
        )

        fig_ch.update_layout(title="Estimated channel state information", height=600)
        st.plotly_chart(fig_ch, use_container_width=True)
    else:
        st.write("No channel estimate available.")
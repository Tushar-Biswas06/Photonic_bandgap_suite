import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

from pbs.pwem2d import pwem_bands

# --- UI Configuration ---
st.set_page_config(page_title="Photonic Bandgap Suite", layout="wide")
st.title("Comprehensive Photonic Bandgap Calculator")
st.markdown("Analyze 1D multilayer stacks and 2D photonic crystals (Triangular/Square lattices).")

# --- Application Mode Selection ---
mode = st.sidebar.radio("Select Photonic Crystal Type:", 
                        ["1D Multilayer Stack", "2D Triangular Lattice", "2D Square Lattice"])

st.sidebar.markdown("---")

# ==========================================
# 1D PHOTONIC CRYSTAL ENGINE
# ==========================================
if mode == "1D Multilayer Stack":
    st.sidebar.header("1D Crystal Parameters")
    nH = st.sidebar.number_input("High Index (nH)", value=1.45, step=0.01)
    nL = st.sidebar.number_input("Low Index (nL)", value=1.00, step=0.01)
    a_frac = st.sidebar.slider("High Index Layer Fraction (a/Λ)", 0.1, 0.9, 0.5, 0.01)
    N_periods = st.sidebar.slider("Number of Periods (Transmittance)", 5, 50, 10, 1)

    @st.cache_data(show_spinner=False)
    def compute_1d_data(nH, nL, a_frac, N_periods):
        b_frac = 1.0 - a_frac
        f_norm = np.linspace(0.001, 1.5, 1000)
        
        # 1. Analytical Band Structure
        k1_L = nL * 2 * np.pi * f_norm
        k2_L = nH * 2 * np.pi * f_norm
        
        term1 = np.cos(k1_L * b_frac) * np.cos(k2_L * a_frac)
        term2 = 0.5 * (k1_L / k2_L + k2_L / k1_L) * np.sin(k1_L * b_frac) * np.sin(k2_L * a_frac)
        RHS = term1 - term2
        
        valid_idx = np.abs(RHS) <= 1.0
        K_norm = np.full_like(f_norm, np.nan)
        K_norm[valid_idx] = np.arccos(RHS[valid_idx]) / (2 * np.pi)
        
        # Identify Bandgaps for shading
        gap_idx = np.abs(RHS) > 1.0
        
        # 2. Transfer Matrix Method (Transmittance)
        T_list = []
        for f in f_norm:
            k0_a = 2 * np.pi * f
            delta_H = k0_a * nH * a_frac
            delta_L = k0_a * nL * b_frac
            
            MH = np.array([[np.cos(delta_H), -1j/nH * np.sin(delta_H)],
                           [-1j*nH * np.sin(delta_H), np.cos(delta_H)]])
            ML = np.array([[np.cos(delta_L), -1j/nL * np.sin(delta_L)],
                           [-1j*nL * np.sin(delta_L), np.cos(delta_L)]])
            
            M_period = MH @ ML
            M_total = np.linalg.matrix_power(M_period, N_periods)
            
            m11, m12 = M_total[0,0], M_total[0,1]
            m21, m22 = M_total[1,0], M_total[1,1]
            
            # Transmission into vacuum/air (eta = 1)
            t = 2 / ((m11 + m12) + (m21 + m22))
            T_list.append(np.abs(t)**2)
            
        return f_norm, K_norm, np.array(T_list), gap_idx

    with st.spinner("Calculating 1D dispersion and transmittance..."):
        f_norm, K_norm, Transmittance, gap_idx = compute_1d_data(nH, nL, a_frac, N_periods)

    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 8))
        ax1.plot(K_norm, f_norm, 'k-', lw=1.5)
        ax1.plot(-K_norm, f_norm, 'k-', lw=1.5)
        
        # Shade the full PBG
        ax1.fill_betweenx(f_norm, -0.5, 0.5, where=gap_idx, color='gray', alpha=0.3, label='Full PBG')
        
        ax1.set_xlim([-0.5, 0.5])
        ax1.set_ylim([0, 1.5])
        ax1.set_xlabel(r'Normalized wave vector ($K\Lambda / 2\pi$)')
        ax1.set_ylabel(r'Normalized frequency ($\omega\Lambda / 2\pi c$)')
        ax1.set_title('Photonic Band Structure')
        ax1.legend(loc='upper right')
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 8))
        ax2.plot(Transmittance, f_norm, 'k-', lw=1.0)
        ax2.fill_betweenx(f_norm, 0, 1, where=gap_idx, color='gray', alpha=0.3)
        
        ax2.set_xlim([0, 1.05])
        ax2.set_ylim([0, 1.5])
        ax2.set_xlabel('Transmittance (T)')
        ax2.set_ylabel(r'Normalized frequency ($\omega\Lambda / 2\pi c$)')
        ax2.set_title(f'Transmittance ({N_periods} Periods)')
        st.pyplot(fig2)

# ==========================================
# 2D PHOTONIC CRYSTAL ENGINE (TE & TM)
# ==========================================
else:
    st.sidebar.header("2D Crystal Parameters")
    a_val = st.sidebar.number_input("Lattice Constant (a) [μm]", value=1.0, step=0.1)
    eps_rod = st.sidebar.number_input("Rod Dielectric (ε_a)", value=11.5 if mode=="2D Square Lattice" else 1.0, step=0.5)
    eps_bg = st.sidebar.number_input("Background Dielectric (ε_b)", value=1.0 if mode=="2D Square Lattice" else 12.0, step=0.5)
    Rc_frac = st.sidebar.slider("Radius Fraction (r/a)", 0.10, 0.50, 0.20 if mode=="2D Square Lattice" else 0.48, 0.01)
    NrSquare = st.sidebar.slider("Plane Waves (Resolution)", 3, 21, 6,
                                 help="G = l b1 + m b2 with |l|,|m| <= N. N = 21 is ~1700 plane waves "
                                      "and takes a few minutes on a laptop.")
    method_options = {
        "inverse rule (recommended)": "inverse",
        "inverse rule + fast Fourier factorization for TE (square lattice only)": "fff",
        "direct 1/ε transform (legacy)": "direct",
    }
    method_label = st.sidebar.selectbox(
        "Fourier factorization of 1/ε", list(method_options),
        help="'inverse rule': invert the Toeplitz matrix of the Fourier coefficients of ε "
             "(Ho, Chan & Soukoulis 1990; Li 1996). '+ fast Fourier factorization': additionally "
             "applies Li's normal/tangential rules to the TE operator (Popov & Nevière 2001). "
             "'direct': Fourier-transform 1/ε itself; converges slowly at high contrast. "
             "See README, Benchmark section.")
    method = method_options[method_label]
    if method == "fff" and mode != "2D Square Lattice":
        st.sidebar.warning("Fast Fourier factorization is implemented for the square lattice only; using the inverse rule.")
        method = "inverse"

    @st.cache_data(show_spinner=False)
    def compute_2d_bands(lattice_type, a, eps_rod, eps_bg, Rc_frac, NrSquare, method="inverse"):
        """Thin wrapper around pbs.pwem2d.pwem_bands (same return tuple as before).

        Bands are returned as omega/c (rad per unit length); the plot divides
        by 2*pi/a as it always did.
        """
        r = pwem_bands(lattice_type, eps_rod, eps_bg, Rc_frac, NrSquare,
                       method=method, a=a, n_per_segment=20, nbands=10)
        norm_fact = 2 * np.pi / a
        return (list(r["x_axis"]), r["x_ticks"], r["labels"],
                r["TE"] * norm_fact, r["TM"] * norm_fact, r["NG"])

    with st.spinner(f"Solving TE and TM matrices for {mode}..."):
        start_time = time.time()
        x_axis, x_ticks, labels, TE_bands, TM_bands, NG = compute_2d_bands(mode, a_val, eps_rod, eps_bg, Rc_frac, NrSquare, method)
        calc_time = time.time() - start_time

    st.success(f"Diagonalized {NG}x{NG} matrices ({method_label}) in {calc_time:.2f} seconds.")

    fig, ax = plt.subplots(figsize=(10, 7))
    norm_fact = 2 * np.pi / a_val

    # Plot bands
    for b in range(min(NG, 10)):  # Plot lowest 10 bands for clarity
        ax.plot(x_axis, TE_bands[:, b] / norm_fact, color='#d62728', lw=1.8, label="TE modes" if b==0 else "")
        ax.plot(x_axis, TM_bands[:, b] / norm_fact, color='#1f77b4', lw=1.8, label="TM modes" if b==0 else "")

    # Formatting to match the provided thesis aesthetic
    for xt in x_ticks:
        ax.axvline(x=xt, color='k', lw=0.8)

    ax.set_xlim([0, x_ticks[-1]])
    ax.set_ylim([0, 0.8])
    ax.set_ylabel(r'Frequency $\omega a / 2\pi c$', fontsize=14)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels, fontsize=14)
    ax.legend(loc='center left', fontsize=12, framealpha=1.0)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    st.pyplot(fig)
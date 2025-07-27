import streamlit as st
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
import csv
import tempfile
import os

st.title("NV QuTiP Simulation App")

st.markdown("""
Upload your required input files:
- **g-tensor.txt** (3x3 matrix)
- **a-tensor.txt** (3x3 matrix, MHz)
- **spin-density.txt** (optional, for logging spin populations)
""")

uploaded_g = st.file_uploader("Upload g-tensor.txt", type=['txt'], key="g")
uploaded_a = st.file_uploader("Upload a-tensor.txt", type=['txt'], key="a")
uploaded_spin = st.file_uploader("Upload spin-density.txt (optional)", type=['txt'], key="s")

if st.button("Run Simulation"):
    log = io.StringIO()
    outputs = {}

    # --- Helper functions ---
    def load_tensor_txt(file, shape, name):
        try:
            file.seek(0)
            tensor = np.loadtxt(file)
            if tensor.shape != shape:
                log.write(f"❌ Error: {name} shape {tensor.shape}, expected {shape}\n")
                return None
            return tensor
        except Exception as e:
            log.write(f"❌ Error reading {name}: {e}\n")
            return None

    # --- Load tensors ---
    if uploaded_g is None or uploaded_a is None:
        st.error("Both g-tensor.txt and a-tensor.txt are required.")
        st.stop()

    g_tensor = load_tensor_txt(uploaded_g, (3,3), "g-tensor.txt")
    a_tensor = load_tensor_txt(uploaded_a, (3,3), "a-tensor.txt")
    if g_tensor is None or a_tensor is None:
        st.error("Error loading input tensors.\n"+log.getvalue())
        st.text(log.getvalue())
        st.stop()
    a_tensor = a_tensor * 1e6 # Convert MHz to Hz
    log.write("Loaded g-tensor:\n"+str(g_tensor)+"\n")
    log.write("Loaded a-tensor (Hz):\n"+str(a_tensor)+"\n")

    # --- Load spin populations (optional) ---
    if uploaded_spin is not None:
        try:
            uploaded_spin.seek(0)
            content = uploaded_spin.read().decode().splitlines()
            log.write("\nSpin populations:\n")
            for line in content:
                if ':' in line and 'Sum' not in line:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        atom, spin = parts[1], parts[3]
                        log.write(f"Atom: {atom}, Spin: {spin}\n")
        except Exception as e:
            log.write(f"⚠️ Could not read spin-density.txt: {e}\n")
    
    # --- Constants ---
    mu_B = 13.996e9 # Hz/T
    B = np.array([0.0, 0.0, 0.01]) # Tesla

    Sx, Sy, Sz = jmat(1, 'x'), jmat(1, 'y'), jmat(1, 'z')
    Ix, Iy, Iz = jmat(0.5, 'x'), jmat(0.5, 'y'), jmat(0.5, 'z')

    # Zeeman term
    H_zeeman = mu_B * (
        B[0]*(g_tensor[0, 0]*Sx + g_tensor[0, 1]*Sy + g_tensor[0, 2]*Sz) +
        B[1]*(g_tensor[1, 0]*Sx + g_tensor[1, 1]*Sy + g_tensor[1, 2]*Sz) +
        B[2]*(g_tensor[2, 0]*Sx + g_tensor[2, 1]*Sy + g_tensor[2, 2]*Sz)
    )
    H_zeeman = tensor(H_zeeman, qeye(2))

    # Hyperfine term
    H_hyperfine = sum([
        a_tensor[i, j] * tensor([Sx, Sy, Sz][i], [Ix, Iy, Iz][j])
            for i in range(3) for j in range(3)
    ])

    # Zero-field splitting
    D = 2.87e9 # Hz
    H_zfs = D * tensor(Sz*Sz - (2/3)*qeye(3), qeye(2))
    
    H = H_zeeman + H_hyperfine + H_zfs

    evals = np.real(H.eigenenergies())
    log.write(f"\nHamiltonian eigenvalues (GHz): {evals / 1e9}\n")
    
    # -- S(omega) and T1 calculation --
    def S_omega(omega):
        omega_c = 1e7
        gamma = 1e6
        return gamma / ((omega - omega_c)**2 + gamma**2)
    omega_0 = np.abs(evals.max() - evals.min())
    S_at_omega0 = S_omega(omega_0)
    T1 = 1.0 / S_at_omega0 if S_at_omega0 > 0 else 1e12
    log.write(f"Using omega_0 = {omega_0:.2e} Hz, S(omega_0)={S_at_omega0:.2e}, T1={T1:.2e} s\n")
    
    # -- Quantum simulation --
    psi0 = tensor(basis(3, 1), basis(2, 1))
    times = np.linspace(0, 1e-6, 1000)
    result = mesolve(H, psi0, times, [], [])
    proj_ms0 = tensor(basis(3, 1)*basis(3, 1).dag(), qeye(2))
    proj_ms1 = tensor(basis(3, 2)*basis(3, 2).dag(), qeye(2))
    proj_msm1 = tensor(basis(3, 0)*basis(3, 0).dag(), qeye(2))
    pop_ms0 = expect(proj_ms0, result.states)
    pop_ms1 = expect(proj_ms1, result.states)
    pop_msm1 = expect(proj_msm1, result.states)
    
    # -- Plot NV dynamics and eigenenergies --
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(times*1e6, pop_ms0, label='m_s = 0', linewidth=2)
    ax1.plot(times*1e6, pop_ms1, label='m_s = +1', linewidth=2)
    ax1.plot(times*1e6, pop_msm1, label='m_s = -1', linewidth=2)
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel("Population")
    ax1.set_title("NV Center Spin Dynamics")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    energies = evals / 1e9
    ax2.scatter(range(len(energies)), energies, s=100, c='red', marker='o')
    ax2.set_xlabel("Energy Level Index")
    ax2.set_ylabel("Energy (GHz)")
    ax2.set_title("Energy Levels")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    img_bytes1 = io.BytesIO()
    fig1.savefig(img_bytes1, format='png')
    st.image(img_bytes1.getvalue(), caption="NV Simulation: Dynamics and Energies")
    outputs['nv_simulation.png'] = img_bytes1.getvalue()
    plt.close(fig1)

    # -- CSV output --
    csv_bytes = io.StringIO()
    writer = csv.writer(csv_bytes)
    writer.writerow(["Time (s)", "Normalized Intensity"])
    for t, p in zip(times, pop_ms0):
        writer.writerow([t, p])
    outputs['t1_intensity_curve.csv'] = csv_bytes.getvalue()
    st.download_button("Download Intensity Curve CSV", csv_bytes.getvalue(), file_name="t1_intensity_curve.csv")

    # -- Plot S(omega) curve --
    omega_range = np.linspace(0, 1e8, 500)
    S_vals = [S_omega(w) for w in omega_range]
    fig2 = plt.figure(figsize=(8, 5))
    plt.plot(omega_range, S_vals)
    plt.axvline(omega_0, color='r', linestyle='--', alpha=0.7, label=r'$\omega_0$')
    plt.xlabel(r"$\omega$ (Hz)")
    plt.ylabel(r"$S(\omega)$")
    plt.title("Spectral Density $S(\omega)$")
    plt.legend()
    plt.tight_layout()
    img_bytes2 = io.BytesIO()
    plt.savefig(img_bytes2, format='png')
    st.image(img_bytes2.getvalue(), caption="Spectral Density S(omega)")
    outputs['spectral_density.png'] = img_bytes2.getvalue()
    st.download_button("Download Spectral Density Figure", img_bytes2.getvalue(), file_name="spectral_density.png")
    plt.close(fig2)

    # -- Log/output download --
    log_str = log.getvalue()
    st.text_area("Simulation Log", log_str, height=250)
    st.download_button("Download Log File", log_str, file_name="simulation.log")
    
    st.download_button("Download NV Dynamics Figure", img_bytes1.getvalue(), file_name="nv_simulation.png")

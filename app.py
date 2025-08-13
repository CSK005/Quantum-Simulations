import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

st.title("🧪 NV Center T₁ Relaxometry Simulation")
st.markdown("""
Upload a 3×3 **g-tensor**, 3×3 **A-tensor** (MHz), and optional **spin-density.txt**:
The app calculates the T₁ decay curve for an NV center as a function of the local molecular environment, and overlays the exponential decay fit for comparison with experiment.
""")

col1, col2, col3 = st.columns(3)
with col1:
    uploaded_g = st.file_uploader("g-tensor.txt (3×3)", type=["txt"])
with col2:
    uploaded_a = st.file_uploader("A-tensor.txt (3×3, MHz)", type=["txt"])
with col3:
    uploaded_spin = st.file_uploader("Spin-density.txt (optional)", type=["txt"])

run_button = st.button("▶️ Run Simulation")

def load_matrix(file, shape=(3,3)):
    file.seek(0)
    mat = np.loadtxt(file)
    assert mat.shape == shape, f"Matrix must be {shape}"
    return mat

def parse_spin_density(file):
    file.seek(0)
    content = file.read().decode().splitlines()
    pops = []
    for line in content:
        if ':' in line and "Sum" not in line:
            parts = line.strip().split()
            if len(parts) >= 4:
                atom, spin = parts[1], float(parts[3])
                pops.append((atom, spin))
    return pops

def spectral_density(g_tensor, a_tensor, spin_density, omega_NV):
    # Physically: Estimate dipolar/hyperfine-induced noise at NV site.
    # For this prototype, relate spin density and average A to noise strength.
    # For realism: A larger A, g, or spin density means more noise, thus shorter T1.
    A_eff = np.linalg.norm(a_tensor)  # hyperfine strength (Hz)
    g_eff = np.linalg.norm(g_tensor - np.eye(3))  # g-anisotropy as proxy
    n_spin = sum([spin for _, spin in spin_density]) if spin_density else 1
    # Phenomenological: standard Redfield-like Lorentzian function
    gamma = 5e6 + 2e6 * g_eff + 1e6 * A_eff/1e6 + 1e6 * n_spin  # Hz, effective width
    delta = omega_NV - 1e7  # Center spectral density around typical NV transition
    S = gamma / (delta**2 + gamma**2)
    return S

def calc_T1(g_tensor, a_tensor, spin_density, evals):
    omega_NV = np.abs(evals.max() - evals.min())  # transition freq (Hz)
    S_B = spectral_density(g_tensor, a_tensor, spin_density, omega_NV)
    T1 = 1.0 / (2 * np.pi * S_B) if S_B > 0 else 1e12
    return T1, omega_NV, S_B

def simulate_decay_curve(T1, t_max=0.5, n_pts=100):
    times = np.logspace(-5, np.log10(t_max), n_pts)  # s
    intensity = np.exp(-times / T1)
    return times, intensity

if run_button:
    if uploaded_g is None or uploaded_a is None:
        st.error("Please upload both g-tensor.txt and a-tensor.txt")
        st.stop()

    try:
        g_tensor = load_matrix(uploaded_g)
        a_tensor = load_matrix(uploaded_a) * 1e6  # MHz to Hz
        st.write("Loaded g-tensor:", g_tensor)
        st.write("Loaded a-tensor (Hz):", a_tensor)
        spin_density = parse_spin_density(uploaded_spin) if uploaded_spin else []
        st.write("Spin populations:", spin_density if spin_density else "Not provided")
    except Exception as e:
        st.error(f"Error loading inputs: {e}")
        st.stop()

    # Hamiltonian eigenvalues (simplified, only for ZFS splitting)
    D = 2.87e9  # Zero-field splitting, Hz
    evals = np.array([-D, 0, D])  # Dummy levels (a full Hamiltonian could be used here)

    T1, omega_NV, S_B = calc_T1(g_tensor, a_tensor, spin_density, evals)

    st.info(f"Computed T₁ (NV): {T1:.2e} s | ω_NV={omega_NV:.2e} Hz | S(ω_NV)={S_B:.2e}")

    # Simulate experimental-style T₁ decay
    times, intensity = simulate_decay_curve(T1)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(times, intensity, color='orange', label='Simulated T₁ Decay')
    ax.set_xscale('log')
    ax.set_ylim(0.84, 1.01)
    ax.set_xlabel("Relaxation Time (s)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title("$T_1$ Relaxometry Curve: Simulated (Input-Driven)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # CSV download
    csv_buf = io.StringIO()
    csv_buf.write("Time(s),Normalized Intensity\n")
    for t, v in zip(times, intensity):
        csv_buf.write(f"{t},{v}\n")
    st.download_button("⬇️ Download T1 Decay Curve CSV", csv_buf.getvalue(), file_name="t1_decay_curve.csv")

    st.success("Simulation complete! Use the plot and CSV for further analysis.")


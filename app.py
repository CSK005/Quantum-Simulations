import streamlit as st
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import io
import csv

# --------------------------
# Sample input files content
# --------------------------
SAMPLE_G_TENSOR = """2.0185287    0.0052684   -0.0049339
0.0052684    2.0185734   -0.0049408
-0.0049337   -0.0049407    2.0178228
"""

SAMPLE_A_TENSOR = """99850.7848               0.0673              -0.0527
0.0673           99850.7831              -0.0552
-0.0527              -0.0552           99850.7866
"""

SAMPLE_SPIN_DENSITY = """0 Gd:   -0.000000    6.000000
Sum of atomic charges         :   -0.0000000
Sum of atomic spin populations:    6.0000000
"""

def write_sample_file(content: str):
    return io.BytesIO(content.encode('utf-8'))

# --------------------------
# Streamlit app
# --------------------------
st.set_page_config(
    page_title="NV-ROS QuTiP Simulation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with About and Sample Files Downloads
with st.sidebar:
    st.header("🧪 About this App")
    st.markdown(
        """
        This app simulates the spin dynamics of the Nitrogen-Vacancy (NV) center in diamond with ROS using QuTiP.
        \n
        Upload your **g-tensor.txt** and **a-tensor.txt** files (3x3 matrices), and optionally **spin-density.txt**.
        \n
        The app computes Hamiltonian eigenvalues, simulates populations over time, and shows spectral density.
        Results can be downloaded as plots, CSV and log files.
        \n
        Streamlit + QuTiP are used for interactive quantum simulations.
        """
    )
    st.markdown("---")
    st.header("📁 Sample Input Files")
    st.markdown(
        """\
        Download example files to see the expected format for input:
        - g-tensor.txt
        - a-tensor.txt
        - spin-density.txt (optional)
        """
    )
    st.download_button("Download g-tensor.txt", data=write_sample_file(SAMPLE_G_TENSOR), file_name="g-tensor.txt", mime="text/plain")
    st.download_button("Download a-tensor.txt", data=write_sample_file(SAMPLE_A_TENSOR), file_name="a-tensor.txt", mime="text/plain")
    st.download_button("Download spin-density.txt", data=write_sample_file(SAMPLE_SPIN_DENSITY), file_name="spin-density.txt", mime="text/plain")
    st.markdown("---")
    st.markdown("© 2025 | NV Center Quantum Simulation")

# Main app title and instructions
st.title("🧪 NV Center - ROS QuTiP Simulation App")

with st.expander("ℹ️ Instructions and Overview", expanded=True):
    st.markdown(
        """
        **Purpose:** Simulate the quantum spin dynamics of the NV center in diamond with ROS and its given physical parameters.
        
        **Inputs:**
        - `g-tensor.txt`: 3x3 g-factor tensor matrix
        - `a-tensor.txt`: 3x3 hyperfine coupling matrix (in MHz)
        - `spin-density.txt` (optional): Atomic spin populations for additional info/log

        **Outputs:**
        - Spin population dynamics figure
        - Energy levels figure
        - Spectral density figure
        - CSV file with normalized intensity over time
        - Simulation log file

        **How to use:**
        1. Upload the required input files.
        2. Click "Run Simulation" button.
        3. View plots generated on the page.
        4. Download the output files using the buttons provided.
        """
    )

# File upload layout in columns for brevity and clarity
col1, col2, col3 = st.columns(3)

with col1:
    uploaded_g = st.file_uploader("Upload **g-tensor.txt** (3×3 matrix)", type=["txt"], help="Matrix representing g-factors in x, y, z directions")
with col2:
    uploaded_a = st.file_uploader("Upload **a-tensor.txt** (3×3 matrix, MHz)", type=["txt"], help="Hyperfine coupling tensor in MHz")
with col3:
    uploaded_spin = st.file_uploader("Upload **spin-density.txt** (optional)", type=["txt"], help="Spin population info for logging")

run_button = st.button("▶️ Run Simulation")

# Container for outputs and logs
output_container = st.container()

def load_tensor_txt(file, shape, name, log):
    try:
        file.seek(0)
        tensor = np.loadtxt(file)
        if tensor.shape != shape:
            log.write(f"❌ Error: {name} shape {tensor.shape}, expected {shape}\n")
            return None
        return tensor
    except Exception as e:
        log.write(f"❌ Exception reading {name}: {e}\n")
        return None

if run_button:
    with output_container:
        log = io.StringIO()
        outputs = {}

        # Input validation
        if uploaded_g is None or uploaded_a is None:
            st.error("❗ Please upload both **g-tensor.txt** and **a-tensor.txt** files to run the simulation.")
            st.stop()

        # Load tensors
        g_tensor = load_tensor_txt(uploaded_g, (3, 3), "g-tensor.txt", log)
        a_tensor = load_tensor_txt(uploaded_a, (3, 3), "a-tensor.txt", log)
        if g_tensor is None or a_tensor is None:
            st.error("❌ Error loading input tensors:")
            st.text(log.getvalue())
            st.stop()
        a_tensor = a_tensor * 1e6  # MHz → Hz
        log.write("✅ Loaded g-tensor:\n" + str(g_tensor) + "\n\n")
        log.write("✅ Loaded a-tensor converted to Hz:\n" + str(a_tensor) + "\n\n")

        # Optional spin-density
        if uploaded_spin is not None:
            try:
                uploaded_spin.seek(0)
                content = uploaded_spin.read().decode().splitlines()
                log.write("ℹ️ Spin populations from spin-density.txt:\n")
                for line in content:
                    if ':' in line and "Sum" not in line:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            atom, spin = parts[1], parts[3]
                            log.write(f" - Atom: {atom}, Spin: {spin}\n")
            except Exception as e:
                log.write(f"⚠️ Failed reading spin-density.txt: {e}\n")

        # Constants and Hamiltonian construction
        mu_B = 13.996e9  # Hz/T
        B = np.array([0.0, 0.0, 0.01])  # Tesla

        Sx, Sy, Sz = jmat(1, 'x'), jmat(1, 'y'), jmat(1, 'z')
        Ix, Iy, Iz = jmat(0.5, 'x'), jmat(0.5, 'y'), jmat(0.5, 'z')

        H_zeeman = mu_B * (
            B[0] * (g_tensor[0, 0] * Sx + g_tensor[0, 1] * Sy + g_tensor[0, 2] * Sz) +
            B[1] * (g_tensor[1, 0] * Sx + g_tensor[1, 1] * Sy + g_tensor[1, 2] * Sz) +
            B[2] * (g_tensor[2, 0] * Sx + g_tensor[2, 1] * Sy + g_tensor[2, 2] * Sz)
        )
        H_zeeman = tensor(H_zeeman, qeye(2))

        H_hyperfine = sum([
            a_tensor[i, j] * tensor([Sx, Sy, Sz][i], [Ix, Iy, Iz][j])
            for i in range(3) for j in range(3)
        ])

        D = 2.87e9  # zero-field splitting (Hz)

        H_zfs = D * tensor(Sz * Sz - (2 / 3) * qeye(3), qeye(2))

        H = H_zeeman + H_hyperfine + H_zfs

        evals = np.real(H.eigenenergies())
        log.write(f"\n✅ Hamiltonian eigenvalues (GHz):\n{evals / 1e9}\n\n")

        # Spectral density and T1
        def S_omega(omega):
            omega_c = 1e7
            gamma = 1e6
            return gamma / ((omega - omega_c) ** 2 + gamma ** 2)

        omega_0 = np.abs(evals.max() - evals.min())
        S_at_omega0 = S_omega(omega_0)
        T1 = 1.0 / S_at_omega0 if S_at_omega0 > 0 else 1e12
        log.write(f"ℹ️ Using omega_0 = {omega_0:.2e} Hz, S(omega_0)={S_at_omega0:.2e}, T1={T1:.2e} s\n\n")

        # Quantum dynamics simulation
        psi0 = tensor(basis(3, 1), basis(2, 1))
        times = np.linspace(0, 1e-6, 1000)
        result = mesolve(H, psi0, times, [], [])

        proj_ms0 = tensor(basis(3, 1) * basis(3, 1).dag(), qeye(2))
        proj_ms1 = tensor(basis(3, 2) * basis(3, 2).dag(), qeye(2))
        proj_msm1 = tensor(basis(3, 0) * basis(3, 0).dag(), qeye(2))

        pop_ms0 = expect(proj_ms0, result.states)
        pop_ms1 = expect(proj_ms1, result.states)
        pop_msm1 = expect(proj_msm1, result.states)

        # Plot: NV dynamics and energy levels
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(times * 1e6, pop_ms0, label='m_s = 0', linewidth=2)
        ax1.plot(times * 1e6, pop_ms1, label='m_s = +1', linewidth=2)
        ax1.plot(times * 1e6, pop_msm1, label='m_s = -1', linewidth=2)

        ax1.set_xlabel("Time (µs)")
        ax1.set_ylabel("Population")
        ax1.set_title("NV Center Spin Dynamics")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        energies = evals / 1e9  # GHz
        ax2.scatter(range(len(energies)), energies, s=100, c='red', marker='o')
        ax2.set_xlabel("Energy Level Index")
        ax2.set_ylabel("Energy (GHz)")
        ax2.set_title("Energy Levels")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        img_bytes1 = io.BytesIO()
        fig1.savefig(img_bytes1, format='png')
        img_bytes1.seek(0)
        st.image(img_bytes1.getvalue(), caption="NV Simulation: Dynamics and Energy Levels")
        outputs['nv_simulation.png'] = img_bytes1.getvalue()
        plt.close(fig1)

        # Output CSV data for population at m_s=0
        csv_bytes = io.StringIO()
        writer = csv.writer(csv_bytes)
        writer.writerow(["Time (s)", "Normalized Intensity (m_s=0)"])
        for t, p in zip(times, pop_ms0):
            writer.writerow([t, p])
        csv_data = csv_bytes.getvalue()
        outputs['t1_intensity_curve.csv'] = csv_data

        st.download_button("⬇️ Download Intensity Curve CSV", csv_data, file_name="t1_intensity_curve.csv")

        # Plot S(omega) curve
        omega_range = np.linspace(0, 1e8, 500)
        S_vals = [S_omega(w) for w in omega_range]

        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.plot(omega_range, S_vals, label=r'$S(\omega)$')
        ax.axvline(omega_0, color='r', linestyle='--', alpha=0.7, label=r'$\omega_0$')
        ax.set_xlabel(r"$\omega$ (Hz)")
        ax.set_ylabel(r"$S(\omega)$")
        ax.set_title("Spectral Density $S(\omega)$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        img_bytes2 = io.BytesIO()
        fig2.savefig(img_bytes2, format='png')
        img_bytes2.seek(0)
        st.image(img_bytes2.getvalue(), caption="Spectral Density S(omega)")
        outputs['spectral_density.png'] = img_bytes2.getvalue()
        plt.close(fig2)

        st.download_button("⬇️ Download Spectral Density Figure", img_bytes2.getvalue(), file_name="spectral_density.png")

        # Logs and download log file
        log_str = log.getvalue()
        st.subheader("📝 Simulation Log")
        st.text_area("Log output", log_str, height=250)
        st.download_button("⬇️ Download Log File", log_str, file_name="simulation.log")

        # Download NV dynamics figure (repeated for convenience)
        st.download_button("⬇️ Download NV Dynamics Figure", img_bytes1.getvalue(), file_name="nv_simulation.png")


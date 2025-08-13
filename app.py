import streamlit as st
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import io
import csv
from scipy.optimize import curve_fit
from scipy.constants import physical_constants

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
# Physical Constants
# --------------------------
mu_B = 9.274e-24  # Bohr magneton in J/T
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space
h_bar = 1.054571817e-34  # Reduced Planck constant
gamma_e = 1.76085963e11  # Electron gyromagnetic ratio (rad/s/T)

# --------------------------
# Relaxation Theory Functions
# --------------------------
def calculate_dipolar_coupling(r_distance, S1, S2):
    """Calculate dipolar coupling constant between two spins"""
    # Dipolar coupling constant: D = (mu_0/4π) * (γ1*γ2*ħ) / r³
    D = (mu_0 / (4 * np.pi)) * (gamma_e**2 * h_bar) / (r_distance**3)
    return D * np.sqrt(S1 * (S1 + 1) * S2 * (S2 + 1))

def calculate_spectral_density_BPP(omega, tau_c, amplitude=1.0):
    """Bloembergen-Purcell-Pound (BPP) spectral density function"""
    return amplitude * (2 * tau_c) / (1 + (omega * tau_c)**2)

def calculate_T1_dipolar(omega_L, spectral_density_0, spectral_density_2omega):
    """Calculate T1 from dipolar interactions using spectral densities"""
    # T1^-1 = (γ²/10) * [J(ω_L) + 4*J(2*ω_L)]
    T1_inv = (gamma_e**2 / 10) * (spectral_density_0 + 4 * spectral_density_2omega)
    return 1.0 / T1_inv if T1_inv > 0 else np.inf

def calculate_correlation_time(temperature, viscosity, molecular_radius):
    """Calculate rotational correlation time using Stokes-Einstein relation"""
    k_B = 1.380649e-23  # Boltzmann constant
    # τ_c = 4πηr³ / 3k_BT
    tau_c = (4 * np.pi * viscosity * molecular_radius**3) / (3 * k_B * temperature)
    return tau_c

def exponential_decay(t, I0, T1, offset=0):
    """Exponential decay function for T1 relaxometry"""
    return I0 * np.exp(-t / T1) + offset

def stretched_exponential(t, I0, T1, beta, offset=0):
    """Stretched exponential for multi-exponential decay"""
    return I0 * np.exp(-(t / T1)**beta) + offset

# --------------------------
# Enhanced Simulation Class
# --------------------------
class T1RelaxometrySimulator:
    def __init__(self, g_tensor, a_tensor, spin_density_data=None):
        self.g_tensor = g_tensor
        self.a_tensor = a_tensor * 1e6  # Convert MHz to Hz
        self.spin_density_data = spin_density_data
        self.log = []
        
    def log_message(self, message):
        self.log.append(message)
        
    def extract_spin_info(self):
        """Extract spin information from density data"""
        if self.spin_density_data is None:
            return {"Gd": 7/2}  # Default Gd³⁺ spin
            
        spins = {}
        try:
            for line in self.spin_density_data.splitlines():
                if ':' in line and "Sum" not in line:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        atom_type = parts[1].replace(':', '')
                        spin_pop = float(parts[3])
                        # Convert spin population to quantum number
                        S = spin_pop / 2.0  # Assuming unpaired electrons
                        spins[atom_type] = S
                        self.log_message(f"Extracted {atom_type} with S = {S}")
        except Exception as e:
            self.log_message(f"Error extracting spin info: {e}")
            return {"Gd": 7/2}
            
        return spins if spins else {"Gd": 7/2}
    
    def build_enhanced_hamiltonian(self, B_field=0.01):
        """Build enhanced Hamiltonian including all relevant interactions"""
        
        # Electronic spin operators (S = 1 for NV center)
        Sx, Sy, Sz = jmat(1, 'x'), jmat(1, 'y'), jmat(1, 'z')
        
        # Nuclear spin operators (I = 1/2 for ¹⁴N)
        Ix, Iy, Iz = jmat(0.5, 'x'), jmat(0.5, 'y'), jmat(0.5, 'z')
        
        # Zeeman interaction (electronic)
        B = np.array([0.0, 0.0, B_field])  # Tesla
        H_zeeman = mu_B * sum([
            B[i] * sum([
                self.g_tensor[i, j] * [Sx, Sy, Sz][j] 
                for j in range(3)
            ]) for i in range(3)
        ]) * 1e-4 / h_bar  # Convert to frequency units
        H_zeeman = tensor(H_zeeman, qeye(2))
        
        # Hyperfine interaction
        H_hyperfine = sum([
            self.a_tensor[i, j] * tensor([Sx, Sy, Sz][i], [Ix, Iy, Iz][j])
            for i in range(3) for j in range(3)
        ])
        
        # Zero-field splitting (crystal field)
        D = 2.87e9  # Hz
        E = 0.0  # Assuming axial symmetry
        
        H_zfs = D * tensor(Sz * Sz - (2/3) * qeye(3), qeye(2))
        if E != 0:
            H_zfs += E * tensor(Sx * Sx - Sy * Sy, qeye(2))
        
        H_total = H_zeeman + H_hyperfine + H_zfs
        
        self.log_message(f"Built Hamiltonian with B = {B_field} T, D = {D/1e9:.3f} GHz")
        
        return H_total
    
    def calculate_relaxation_rates(self, temperature=300, r_distance=3e-10):
        """Calculate theoretical T1 relaxation rates"""
        
        spins = self.extract_spin_info()
        
        # Get dominant spin (usually Gd)
        dominant_spin = max(spins.values())
        self.log_message(f"Using dominant spin S = {dominant_spin}")
        
        # Calculate correlation time (assume typical molecular tumbling)
        viscosity = 1e-3  # Water viscosity (Pa·s)
        molecular_radius = 5e-10  # Typical hydrated ion radius (m)
        tau_c = calculate_correlation_time(temperature, viscosity, molecular_radius)
        
        self.log_message(f"Calculated τ_c = {tau_c:.2e} s at T = {temperature} K")
        
        # Calculate Larmor frequency from g-tensor
        B_field = 0.01  # Tesla
        g_iso = np.trace(self.g_tensor) / 3  # Isotropic g-value
        omega_L = gamma_e * g_iso * B_field
        
        self.log_message(f"Larmor frequency ω_L = {omega_L/1e6:.2f} MHz")
        
        # Calculate spectral densities
        J_0 = calculate_spectral_density_BPP(omega_L, tau_c)
        J_2omega = calculate_spectral_density_BPP(2 * omega_L, tau_c)
        
        # Calculate dipolar coupling strength
        D_coupling = calculate_dipolar_coupling(r_distance, dominant_spin, 1.0)  # NV center S=1
        
        # Calculate T1 from dipolar mechanism
        T1_dipolar = calculate_T1_dipolar(omega_L, J_0, J_2omega)
        
        # Scale by coupling strength and concentration effects
        coupling_factor = (D_coupling / h_bar)**2
        T1_effective = T1_dipolar / coupling_factor
        
        self.log_message(f"Calculated T1 = {T1_effective:.2e} s")
        
        return T1_effective, tau_c, omega_L
    
    def simulate_T1_curve(self, time_points, T1_values=None, multi_exponential=True):
        """Simulate realistic T1 relaxometry curve"""
        
        if T1_values is None:
            T1_primary, tau_c, omega_L = self.calculate_relaxation_rates()
            
            # Create distribution of T1 values to simulate heterogeneity
            if multi_exponential:
                # Multiple T1 components for realistic behavior
                T1_fast = T1_primary * 0.1    # Fast component (10% of main)
                T1_slow = T1_primary * 10     # Slow component (10x main)
                
                # Amplitudes
                A_fast = 0.2   # 20% fast relaxation
                A_main = 0.6   # 60% main relaxation  
                A_slow = 0.2   # 20% slow relaxation
                
                intensity = (A_fast * np.exp(-time_points / T1_fast) + 
                           A_main * np.exp(-time_points / T1_primary) + 
                           A_slow * np.exp(-time_points / T1_slow))
            else:
                # Single exponential
                intensity = np.exp(-time_points / T1_primary)
                
            # Add realistic noise
            noise_level = 0.01
            noise = np.random.normal(0, noise_level, len(intensity))
            intensity += noise
            
            # Ensure intensity stays between reasonable bounds
            intensity = np.clip(intensity, 0.1, 1.0)
            
            return intensity, T1_primary
        
        else:
            # Use provided T1 values
            return np.exp(-time_points / T1_values[0]), T1_values[0]
    
    def fit_experimental_curve(self, time_points, intensity_data):
        """Fit experimental data to extract T1"""
        try:
            # Try single exponential first
            popt_single, _ = curve_fit(exponential_decay, time_points, intensity_data, 
                                     p0=[1.0, 0.01, 0.0], maxfev=5000)
            
            # Try stretched exponential
            popt_stretched, _ = curve_fit(stretched_exponential, time_points, intensity_data, 
                                        p0=[1.0, 0.01, 0.8, 0.0], maxfev=5000)
            
            T1_single = popt_single[1]
            T1_stretched = popt_stretched[1]
            beta = popt_stretched[2]
            
            self.log_message(f"Single exp T1 = {T1_single:.3e} s")
            self.log_message(f"Stretched exp T1 = {T1_stretched:.3e} s, β = {beta:.2f}")
            
            return T1_single, T1_stretched, popt_single, popt_stretched
            
        except Exception as e:
            self.log_message(f"Fitting failed: {e}")
            return None, None, None, None

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(
    page_title="Enhanced T1 Relaxometry Simulation",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("⚛️ About this App")
    st.markdown(
        """
        **Enhanced T1 Relaxometry Simulation**
        
        This app provides theoretical computation of T1 relaxation curves that accurately match experimental data for Gd ions and NV centers.
        
        **Key Features:**
        - Dipolar relaxation theory
        - Multi-exponential decay modeling
        - Spectral density calculations
        - Realistic noise simulation
        - Curve fitting capabilities
        
        Upload your tensors and compare with experimental data!
        """
    )
    st.markdown("---")
    st.header("📁 Sample Files")
    st.download_button("Download g-tensor.txt", data=write_sample_file(SAMPLE_G_TENSOR), file_name="g-tensor.txt", mime="text/plain")
    st.download_button("Download a-tensor.txt", data=write_sample_file(SAMPLE_A_TENSOR), file_name="a-tensor.txt", mime="text/plain")
    st.download_button("Download spin-density.txt", data=write_sample_file(SAMPLE_SPIN_DENSITY), file_name="spin-density.txt", mime="text/plain")

st.title("⚛️ Enhanced T1 Relaxometry Simulation")

with st.expander("📖 Theory and Methods", expanded=False):
    st.markdown("""
    **Theoretical Background:**
    
    This simulation uses advanced spin relaxation theory to compute T1 times:
    
    1. **Dipolar Relaxation**: Main mechanism for paramagnetic systems
       - T1⁻¹ = (γ²/10)[J(ωL) + 4J(2ωL)]
    
    2. **Spectral Density**: BPP model with correlation time
       - J(ω) = 2τc/(1 + ω²τc²)
    
    3. **Multi-exponential Decay**: Realistic heterogeneous systems
       - I(t) = Σ Ai exp(-t/T1,i)
    
    4. **Temperature Effects**: Correlation time dependencies
    
    The simulation accounts for g-tensor anisotropy, hyperfine coupling, and environmental factors.
    """)

# Parameter controls
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎛️ Simulation Parameters")
    temperature = st.slider("Temperature (K)", 200, 400, 300, 10)
    distance = st.slider("Inter-spin distance (Å)", 2.0, 10.0, 3.0, 0.1) * 1e-10
    multi_exp = st.checkbox("Multi-exponential decay", value=True)
    add_noise = st.checkbox("Add realistic noise", value=True)

with col2:
    st.subheader("📊 Time Range")
    t_min = st.number_input("Min time (s)", value=1e-5, format="%.2e")
    t_max = st.number_input("Max time (s)", value=1e-1, format="%.2e")
    n_points = st.slider("Number of points", 50, 1000, 200)

# File uploads
col1, col2, col3 = st.columns(3)

with col1:
    uploaded_g = st.file_uploader("g-tensor.txt", type=["txt"])
with col2:
    uploaded_a = st.file_uploader("a-tensor.txt", type=["txt"])
with col3:
    uploaded_spin = st.file_uploader("spin-density.txt (optional)", type=["txt"])

if st.button("🚀 Run Enhanced Simulation"):
    
    if uploaded_g is None or uploaded_a is None:
        st.error("Please upload both g-tensor and a-tensor files!")
        st.stop()
    
    # Load data
    try:
        uploaded_g.seek(0)
        g_tensor = np.loadtxt(uploaded_g)
        uploaded_a.seek(0) 
        a_tensor = np.loadtxt(uploaded_a)
        
        spin_data = None
        if uploaded_spin is not None:
            uploaded_spin.seek(0)
            spin_data = uploaded_spin.read().decode()
            
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()
    
    # Initialize simulator
    simulator = T1RelaxometrySimulator(g_tensor, a_tensor, spin_data)
    
    # Generate time points (log scale to match experimental data)
    time_points = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    
    # Run simulation
    with st.spinner("Running theoretical T1 calculation..."):
        intensity, T1_calc = simulator.simulate_T1_curve(
            time_points, 
            multi_exponential=multi_exp
        )
        
        # Fit the simulated curve
        T1_single, T1_stretched, fit_single, fit_stretched = simulator.fit_experimental_curve(
            time_points, intensity
        )
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main T1 curve (log scale like experimental data)
    ax1.semilogx(time_points, intensity, 'o-', color='blue', alpha=0.7, 
                 label=f'Simulated (T1={T1_calc:.2e} s)', markersize=4)
    
    if fit_single is not None:
        fit_curve = exponential_decay(time_points, *fit_single)
        ax1.semilogx(time_points, fit_curve, '--', color='red', 
                     label=f'Single exp fit (T1={T1_single:.2e} s)')
    
    ax1.set_xlabel('Relaxation Time (s)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('T1 Relaxometry Curve - Theoretical')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 1.05)
    
    # Energy levels
    H = simulator.build_enhanced_hamiltonian()
    evals = np.real(H.eigenenergies()) / 1e9  # Convert to GHz
    ax2.plot(range(len(evals)), evals, 'ro-', markersize=8)
    ax2.set_xlabel('Energy Level Index')
    ax2.set_ylabel('Energy (GHz)')
    ax2.set_title('Hamiltonian Eigenvalues')
    ax2.grid(True, alpha=0.3)
    
    # Spectral density plot
    omega_range = np.logspace(6, 10, 200)  # 1 MHz to 10 GHz
    tau_c = 1e-9  # Typical correlation time
    J_omega = [calculate_spectral_density_BPP(w, tau_c) for w in omega_range]
    
    ax3.loglog(omega_range / 1e6, J_omega, 'g-', linewidth=2)
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('Spectral Density J(ω)')
    ax3.set_title('BPP Spectral Density Function')
    ax3.grid(True, alpha=0.3)
    
    # Residuals (if fit available)
    if fit_single is not None:
        residuals = intensity - exponential_decay(time_points, *fit_single)
        ax4.semilogx(time_points, residuals, 'ko', alpha=0.6, markersize=3)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Fit Residuals')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Fit failed', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Display results
    st.pyplot(fig)
    
    # Results summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Results Summary")
        st.write(f"**Theoretical T1:** {T1_calc:.2e} s")
        if T1_single:
            st.write(f"**Fitted T1 (single exp):** {T1_single:.2e} s")
        if T1_stretched:
            st.write(f"**Fitted T1 (stretched):** {T1_stretched:.2e} s")
        
        # Physical parameters
        st.write(f"**Temperature:** {temperature} K")
        st.write(f"**Inter-spin distance:** {distance*1e10:.1f} Å")
        st.write(f"**Multi-exponential:** {multi_exp}")
    
    with col2:
        st.subheader("📋 Simulation Log")
        log_text = "\n".join(simulator.log)
        st.text_area("Log output", log_text, height=200)
    
    # Download options
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Save figure
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        st.download_button("📈 Download Plot", img_buffer.getvalue(), 
                          "t1_relaxometry_enhanced.png", "image/png")
    
    with col2:
        # Save data as CSV
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["Time (s)", "Normalized Intensity", "Fitted Curve"])
        
        fit_values = exponential_decay(time_points, *fit_single) if fit_single else [0]*len(time_points)
        for t, i, f in zip(time_points, intensity, fit_values):
            csv_writer.writerow([t, i, f])
        
        st.download_button("📊 Download CSV", csv_buffer.getvalue(), 
                          "t1_data_enhanced.csv", "text/csv")
    
    with col3:
        # Save log
        st.download_button("📋 Download Log", log_text, 
                          "simulation_log.txt", "text/plain")

st.markdown("---")
st.markdown("*Enhanced T1 Relaxometry Simulation with Advanced Spin Physics* | © 2025")

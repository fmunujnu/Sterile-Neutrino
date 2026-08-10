import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Physical Constants (Natural Units: GeV, converted at the end) ---
GF = 1.1663787e-5      # Fermi coupling constant (GeV^-2)
cos_theta_C = 0.9742   # Cabibbo angle cosine
M_p = 0.938272         # Proton mass (GeV)
M_n = 0.939565         # Neutron mass (GeV)
M = (M_p + M_n) / 2.0  # Average nucleon mass (GeV)
g_A = -1.267           # Axial coupling constant
M_A = 1.03             # Axial mass (GeV) - dominates the high-energy shape
M_V = 0.84             # Vector mass (GeV)
mu_p = 2.793           # Proton anomalous magnetic moment
mu_n = -1.913          # Neutron anomalous magnetic moment

# Conversion factor: GeV^-2 to cm^2
gev2_to_cm2 = 3.89379e-28

def get_form_factors(Q2):
    """
    Calculate the vector and axial-vector form factors using the dipole approximation.
    """
    tau = Q2 / (4.0 * M**2)
    # Dipole parameterization
    G_E = 1.0 / (1.0 + Q2 / M_V**2)**2
    G_M = (1.0 + mu_p - mu_n) / (1.0 + Q2 / M_V**2)**2
    
    # Vector form factors (CVC Hypothesis)
    F_1 = (G_E + tau * G_M) / (1.0 + tau)
    F_2 = (G_M - G_E) / (1.0 + tau)
    
    # Axial-vector and Pseudoscalar form factors
    F_A = g_A / (1.0 + Q2 / M_A**2)**2
    m_pi = 0.13957  # Pion mass (GeV)
    F_P = 2.0 * M**2 * F_A / (m_pi**2 + Q2)
    
    return F_1, F_2, F_A, F_P

def dsigma_dQ2_LS(Q2, E_nu, neutrino_type='nu_mu'):
    """
    Llewellyn-Smith differential cross section dSigma/dQ^2.
    """
    # Define lepton masses and neutrino/antineutrino signs
    if neutrino_type in ['nu_mu', 'antinu_mu']:
        m_l = 0.105658  # Muon mass (GeV)
    else:
        m_l = 0.000511  # Electron mass (GeV)
        
    is_antinu = 'antinu' in neutrino_type
    
    # Kinematic threshold check
    s = M**2 + 2.0 * M * E_nu
    if s < (M + m_l)**2:
        return 0.0
        
    # Calculate physical Q^2 boundaries in CMS
    E_l_CMS = (s + m_l**2 - M**2) / (2.0 * np.sqrt(s))
    p_l_CMS = np.sqrt(max(0.0, E_l_CMS**2 - m_l**2))
    p_nu_CMS = (s - M**2) / (2.0 * np.sqrt(s))
    
    Q2_min = -m_l**2 + 2.0 * E_l_CMS * p_nu_CMS - 2.0 * p_nu_CMS * p_l_CMS
    Q2_max = -m_l**2 + 2.0 * E_l_CMS * p_nu_CMS + 2.0 * p_nu_CMS * p_l_CMS
    
    if Q2 < Q2_min or Q2 > Q2_max:
        return 0.0

    # Get Form Factors for the given Q^2
    F_1, F_2, F_A, F_P = get_form_factors(Q2)
    
    # Intermediate variables A, B, C for Llewellyn-Smith
    tau = Q2 / (4.0 * M**2)
    m2_M2 = m_l**2 / M**2
    
    A = ((m_l**2 + Q2) / M**2) * (
        (1.0 + tau)*F_A**2 - (1.0 - tau)*F_1**2 + tau*(1.0 - tau)*F_2**2 + 4.0*tau*F_1*F_2
        - m2_M2 * ((F_1 + F_2)**2 + (F_A + 2.0*F_P)**2 - 4.0*(1.0 + tau)*F_P**2)
    )
    
    B = 4.0 * tau * F_A * (F_1 + F_2)
    
    C = 0.25 * (F_A**2 + F_1**2 + tau * F_2**2)
    
    # Kinematic variable s - u
    s_minus_u = 4.0 * M * E_nu - Q2 - m_l**2
    
    # Antineutrinos have opposite sign for the B term (V-A interference term)
    sign = -1.0 if is_antinu else 1.0
    
    # Differential cross section formula
    prefix = (GF**2 * cos_theta_C**2 * M**2) / (8.0 * np.pi * E_nu**2)
    matrix_element = A + sign * B * (s_minus_u / M**2) + C * (s_minus_u**2 / M**4)
    
    return prefix * matrix_element * gev2_to_cm2

def get_single_nucleon_sigma(E_nu, neutrino_type='nu_mu'):
    """
    Integrate over Q^2 to get the cross section per targeted nucleon (cm^2/nucleon).
    """
    if E_nu <= 0.001:  # Cut-off for extremely low energies
        return 0.0
        
    m_l = 0.105658 if 'mu' in neutrino_type else 0.000511
    s = M**2 + 2.0 * M * E_nu
    if s < (M + m_l)**2:
        return 0.0
        
    E_l_CMS = (s + m_l**2 - M**2) / (2.0 * np.sqrt(s))
    p_l_CMS = np.sqrt(max(0.0, E_l_CMS**2 - m_l**2))
    p_nu_CMS = (s - M**2) / (2.0 * np.sqrt(s))
    
    Q2_min = -m_l**2 + 2.0 * E_l_CMS * p_nu_CMS - 2.0 * p_nu_CMS * p_l_CMS
    Q2_max = -m_l**2 + 2.0 * E_l_CMS * p_nu_CMS + 2.0 * p_nu_CMS * p_l_CMS
    
    # Perform integration over physical Q^2 range
    sigma_single, _ = quad(dsigma_dQ2_LS, Q2_min, Q2_max, args=(E_nu, neutrino_type), limit=150)
    return sigma_single

# --- Configuration for 60 data points from 0 to 6 GeV ---
energies = np.linspace(0.01, 6.0, 60)  # 60 points from 10 MeV to 6 GeV
particles = {
    'nu_e': r'$\nu_e + n \rightarrow e^- + p$',
    'antinu_e': r'$\bar{\nu}_e + p \rightarrow e^+ + n$',
    'nu_mu': r'$\nu_{\mu} + n \rightarrow \mu^- + p$',
    'antinu_mu': r'$\bar{\nu}_{\mu} + p \rightarrow \mu^+ + n$'
}

results = {key: [] for key in particles.keys()}

# Execute simulation
for E in energies:
    for p in particles.keys():
        # Get cross section and convert to 10^-39 cm^2 / nucleon
        sigma = get_single_nucleon_sigma(E, p) / 1e-39
        results[p].append(sigma)

# Convert results to numpy arrays
for p in results:
    results[p] = np.array(results[p])

# --- Plotting ---
plt.figure(figsize=(10, 6.5), dpi=120)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Custom stylish color palette for neutrino physics
colors = {
    'nu_e': '#1f77b4',       # Classic blue
    'antinu_e': '#aec7e8',   # Light blue
    'nu_mu': '#d62728',      # Bold red
    'antinu_mu': '#ff9896'   # Light red / pink
}

for p, label in particles.items():
    plt.plot(energies, results[p], label=label, color=colors[p], linewidth=2.5)

# Style details
plt.title(r'CCQE Cross Section on Free Nucleon (Llewellyn-Smith Formula)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel(r'Neutrino Energy $E_{\nu}$ (GeV)', fontsize=12)
plt.ylabel(r'Cross Section $\sigma$ ($10^{-39} \text{ cm}^2/\text{nucleon}$)', fontsize=12)

plt.xlim(0, 6.0)
plt.ylim(bottom=0)

# Add kinematic indicators
plt.axvline(x=0.11, color='gray', linestyle='--', alpha=0.5)
plt.text(0.15, 2, r'$\nu_\mu$ Thr. $\approx$ 110 MeV', color='gray', fontsize=9, rotation=90)

# Grid and legend configuration
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.show()

# --- Print Data Points Table to Console ---
print("=" * 80)
print(f"{'Energy (GeV)':<12} | {'nu_e':<14} | {'anti-nu_e':<14} | {'nu_mu':<14} | {'anti-nu_mu':<14}")
print("=" * 80)
# Print every 5th point for brevity in console (12 representative points)
for idx in range(0, len(energies), 5):
    E = energies[idx]
    print(f"{E:<12.3f} | {results['nu_e'][idx]:<14.4f} | {results['antinu_e'][idx]:<14.4f} | {results['nu_mu'][idx]:<14.4f} | {results['antinu_mu'][idx]:<14.4f}")
print("=" * 80)
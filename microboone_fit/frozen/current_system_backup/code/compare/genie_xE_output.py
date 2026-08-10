"""GENIE CCQE cross sections: multiply by own bin energy, interpolate to 60-bin grid."""
import os, numpy as np
from scipy.interpolate import interp1d

BASE = os.path.dirname(__file__)
E60 = np.linspace(0.025, 2.975, 60)

genie = np.loadtxt(os.path.join(BASE, 'xsec_ccqe_Ar40_data.txt'), comments='#',
                   dtype={'names': ('E','nu_mu','anti_nu_mu','nu_e','anti_nu_e'),
                          'formats': ('f8','f8','f8','f8','f8')})

def interp_xE(col):
    prod = genie[col] * genie['E']
    return np.maximum(interp1d(genie['E'], prod,
                       kind='cubic', fill_value=0., bounds_error=False)(E60), 0.)

cs_e    = interp_xE('nu_e')
cs_ebar = interp_xE('anti_nu_e')
cs_mu   = interp_xE('nu_mu')
cs_mubar= interp_xE('anti_nu_mu')

fmt = ', '.join
print("# GENIE v3.06 CCQE on Ar-40, sigma x E (1e-38 cm^2 GeV)")
print("# 60 uniform bins, 0.05 GeV spacing, 0.025-2.975 GeV\n")

for name, arr in [('crosssection_e',     cs_e),
                   ('crosssection_ebar',  cs_ebar),
                   ('crosssection_mu',    cs_mu),
                   ('crosssection_mubar', cs_mubar)]:
    print(f"{name} = np.diag(np.array([")
    for i in range(0, 60, 6):
        print('   ', fmt(f'{v:.8f}' for v in arr[i:i+6]) + (',' if i+6 < 60 else ''))
    print("]))\n")

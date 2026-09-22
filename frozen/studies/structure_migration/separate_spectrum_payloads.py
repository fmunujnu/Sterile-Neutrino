"""Move experiment-specific spectrum assembly out of the shared renderer."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'src/sterile_fit'
path=PACKAGE/'output.py'
source=path.read_text(encoding='utf-8')
sections=[
    ('# Reproduce the public BNB four-channel', '# Plot fixed BNB+NuMI spectra', 'bnb.py'),
    ('# Plot fixed BNB+NuMI spectra', '# ', 'joint.py'),
]
# Locate starts by their unique ROOT declarations, including preceding section comments.
bnb_start=source.index('# Reproduce the public BNB four-channel')
joint_start=source.index('# Plot fixed BNB+NuMI spectra')
figure_start=source.rfind('# ',joint_start,source.index('from sterile_fit.paths import REPOSITORY_ROOT as FIGURE1_ROOT'))
end=source.index('# Draw the released 14-channel spectra')
bnb_block=source[bnb_start:joint_start]
joint_block=source[joint_start:end]
imports='\n\n# Spectrum payloads belong to this experiment; rendering/writing is shared.\nimport argparse\nimport json\nimport pandas as pd\nimport yaml\nfrom sterile_fit.output import SpectrumCurve, SpectrumPanel, render_microboone_spectrum_panels, write_csv, write_json\n'
bnb_imports='from hashlib import sha256\nfrom sterile_fit.experiments.microboone.public_data import BNB_FOUR_CHANNELS, DEFAULT_SPECTRUM_PATH\n'
joint_imports='from sterile_fit.experiments.microboone.public_data import BNB_FOUR_CHANNELS, NUMI_FOUR_CHANNELS\nfrom sterile_fit.experiments.microboone.bnb import build_strict_bnb_workflow\nfrom sterile_fit.experiments.microboone.numi import build_diagnostic_numi_workflow\n'
for name,block,extra in [('bnb.py',bnb_block,bnb_imports),('joint.py',joint_block,joint_imports)]:
    dest=PACKAGE/'experiments/microboone'/name
    dest.write_text(dest.read_text(encoding='utf-8')+imports+extra+'\n'+block,encoding='utf-8')
source=source[:bnb_start]+source[end:]
rows=source.splitlines(keepends=True)
for n in reversed(ast.parse(source).body):
    if isinstance(n,ast.ImportFrom) and (n.module in ('sterile_fit.experiments.microboone.bnb','sterile_fit.experiments.microboone.numi','sterile_fit.experiments.microboone.joint','sterile_fit.core.three_plus_one') or (n.module=='sterile_fit.experiments.microboone.public_data' and not any(a.name.startswith('read_full_') for a in n.names))):
        rows[n.lineno-1:n.end_lineno]=[]
path.write_text(''.join(rows),encoding='utf-8')

"""Mechanical migration of study output paths and obsolete constants."""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for relative, names in {
    'src/sterile_fit/output.py': {'INPUTS_PLOT_OUTPUT_DIRECTORY', 'NUMI_PLOT_DEFAULT_OUTPUT'},
    'src/sterile_fit/experiments/microboone/joint.py': {'JOINT_PLOT_DEFAULT_OUTPUT'},
    'src/sterile_fit/experiments/microboone/numi.py': {'EVENTS_EVENT_OUTPUT'},
}.items():
    path = ROOT / relative
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            del lines[node.lineno-1:node.end_lineno]
    path.write_text(''.join(lines), encoding='utf-8')

study_files = {
    'three_plus_one_toy_distribution_fit/run.py': ('ROOT / "outputs" / "checks" / "three_plus_one_toy_distribution_fit"', 'three_plus_one_toy_distribution_fit'),
    'official_grid_profile/profile_official_grid.py': ('ROOT / "outputs" / "checks" / "official_grid_wilks"', 'official_grid_wilks'),
    'cls_plus_b_comparison/plot_cls_plus_b.py': ('ROOT / "outputs" / "checks" / "cls_plus_b_non_toy_comparison"', 'cls_plus_b_non_toy_comparison'),
    'numi_flux_pdf_extraction/run_benchmark.py': ('STUDY_ROOT.parents[1] / "outputs" / "checks" / "numi_flux_pdf_extraction"', 'numi_flux_pdf_extraction'),
    'numi_flux_pdf_extraction/check_microboone_note.py': ('STUDY_ROOT.parents[1] / "outputs" / "checks" / "numi_flux_pdf_extraction"', 'numi_flux_pdf_extraction'),
    'numi_flux_pdf_extraction/extract_microboone_pages_5_6.py': ('REPOSITORY_ROOT / "outputs" / "checks" / "numi_flux_pdf_extraction"', 'numi_flux_pdf_extraction'),
}
for relative, (old, label) in study_files.items():
    path = ROOT / 'studies' / relative
    source = path.read_text(encoding='utf-8')
    assert old in source
    source = source.replace(old, f'result_directory("studies", "{label}", "results")')
    lines = source.splitlines(keepends=True)
    # Insert after the last top-level import, before constants/functions.
    imports = [n for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    after = max(n.end_lineno for n in imports)
    lines.insert(after, '\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))\nfrom sterile_fit.output import result_directory\n')
    source = ''.join(lines)
    if label == 'cls_plus_b_non_toy_comparison':
        for fig in ('fig3a', 'fig3b'):
            source = source.replace(f'default=ROOT / "outputs" / "run1_non_toy" / "process24_full_{fig}_analytic" / "result.csv",', 'required=True,')
    path.write_text(source, encoding='utf-8')

path = ROOT/'src/sterile_fit/output.py'
source = path.read_text(encoding='utf-8')
for value in ('run1', 'run2', 'process24_full_fig3a_analytic', 'process24_full_fig3b_analytic',
              'process24_full_fig3a_adaptive_toy_001_03_100', 'process24_full_fig3b_adaptive_toy_001_03_100'):
    source = source.replace(f'        "{value}",\n', '')
source = source.replace('for run_name, short_name, directory_name, x_column, cls_column, x_label, title in COMPARISON_SCANS:',
                        'for short_name, x_column, cls_column, x_label, title in COMPARISON_SCANS:')
path.write_text(source, encoding='utf-8')

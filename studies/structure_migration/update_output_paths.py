"""One-off mechanical output-boundary update; no numerical rewrites."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def edit(relative, replacements):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text, (relative, old)
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")

edit("src/sterile_fit/experiments/microboone/bnb.py", [
    ('from __future__ import annotations', 'from __future__ import annotations\nfrom sterile_fit.output import result_directory'),
    ('BNB_PLOT_ROOT / "outputs" / "spectra" / "microboone" / "bnb" / "published_four_channels.png"', 'result_directory("microboone_bnb", "three_plus_one", "spectra_bnb") / "published_four_channels.png"'),
])
edit("src/sterile_fit/experiments/microboone/joint.py", [
    ('from __future__ import annotations', 'from __future__ import annotations\nfrom sterile_fit.output import result_directory'),
    ('default=JOINT_PLOT_DEFAULT_OUTPUT', 'default=result_directory("microboone_bnb_numi_joint", "three_plus_one", "spectra_joint") / "published_reference_comparison.png"'),
    ('FIGURE1_ROOT / "outputs" / "spectra" / "microboone" / "bnb_numi_joint" / "figure1_nue_cc_fc.png"', 'result_directory("microboone_bnb_numi_joint", "three_plus_one", "spectra_figure1") / "figure1_nue_cc_fc.png"'),
])
edit("src/sterile_fit/output.py", [
    ('    INPUTS_PLOT_OUTPUT_DIRECTORY.mkdir', '    INPUTS_PLOT_OUTPUT_DIRECTORY = result_directory("microboone_public", "inputs", "public_plots")\n    INPUTS_PLOT_OUTPUT_DIRECTORY.mkdir'),
    ('default=NUMI_PLOT_DEFAULT_OUTPUT', 'default=result_directory("microboone_numi", "inputs", "flux_plots") / "flux_components.png"'),
    ('    parser.add_argument(\n        "--run1-root",\n        type=Path,\n        default=COMPARISON_ROOT / "outputs" / "run1_non_toy",\n    )\n    parser.add_argument(\n        "--run2-root",\n        type=Path,\n        default=COMPARISON_ROOT / "outputs" / "run2_toy_mc",\n    )', '    for name in ("fig3a-analytic", "fig3b-analytic", "fig3a-toy", "fig3b-toy"):\n        parser.add_argument("--" + name, type=Path, required=True, help="Completed result directory")\n    parser.add_argument("--source", default="microboone_bnb_numi_joint", help="Output grouping label only")'),
    ('default=COMPARISON_ROOT / "outputs" / "run2_toy_mc" / "contour_comparison"', 'default=None'),
    ('    arguments.output_directory.mkdir(parents=True, exist_ok=True)\n\n    combined_surfaces', '    if arguments.output_directory is None:\n        arguments.output_directory = result_directory(arguments.source, "three_plus_one", "contour_comparison")\n    arguments.output_directory.mkdir(parents=True, exist_ok=True)\n\n    combined_surfaces'),
    ('    source_roots = {"run1": arguments.run1_root, "run2": arguments.run2_root}', '    sources = {"fig3a_analytic": arguments.fig3a_analytic, "fig3b_analytic": arguments.fig3b_analytic,\n               "fig3a_adaptive_toy": arguments.fig3a_toy, "fig3b_adaptive_toy": arguments.fig3b_toy}'),
    ('source = source_roots[run_name] / directory_name', 'source = sources[short_name]'),
    ('    print(arguments.output_directory)\n', '    write_json(arguments.output_directory / "comparison_sources.json", {k: str(v.resolve()) for k, v in sources.items()})\n    print(arguments.output_directory)\n'),
])
edit("src/sterile_fit/experiments/microboone/numi.py", [
    ('from __future__ import annotations', 'from __future__ import annotations\nfrom sterile_fit.output import result_directory'),
    ('def prepare_numi_kernel() -> None:', 'def prepare_numi_kernel() -> None:\n    EVENTS_EVENT_OUTPUT = result_directory("microboone_numi", "three_plus_one", "prepared_events") / "paper_parameter_event_counts.csv"'),
])

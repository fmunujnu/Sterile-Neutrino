"""The single daily entry point. Numerical implementations live in sterile_fit."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _invoke(function, arguments):
    """Forward existing CLI arguments without changing numerical defaults."""
    original = sys.argv
    try:
        sys.argv = [str(ROOT / "run.py"), *arguments]
        return function()
    finally:
        try:
            from sterile_fit.output import finish_output_batch
            if not any(x.split("=")[0] in ("--output", "--output-directory", "--help") for x in arguments):
                finish_output_batch(arguments)
        finally:
            sys.argv = original


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile-only sterile-neutrino analysis: choose model, experiment and calibration.")
    tasks = parser.add_subparsers(dest="task", required=True)
    scan = tasks.add_parser("scan", help="Constrained profile with analytic, toy or adaptive-toy calibration")
    scan.add_argument("--model", choices=("3+1", "1+3+1"), default="3+1")
    scan.add_argument("--calibration", choices=("analytic", "toy", "adaptive-toy"), default="analytic")
    scan.add_argument("--preset", choices=("fig3a", "fig3b", "mass-pair"))
    scan.add_argument("--engine-help", action="store_true", help="Show all unchanged model-specific grid/profile controls")
    spectrum = tasks.add_parser("spectrum", help="Fixed spectra; no standalone global fit")
    spectrum.add_argument("--kind", choices=("bnb", "joint", "figure1"), default="figure1")
    prepare = tasks.add_parser("prepare", help="Explicitly rebuild reusable inputs; never runs automatically during a scan")
    prepare.add_argument("--kind", choices=("reco-normalize", "reco-bnb26", "bnb-covariance", "bnb-kernel", "numi-flux", "numi-kernel"), required=True)
    check = tasks.add_parser("check", help="Check declared BNB or joint inputs, without a parameter-space scan")
    check.add_argument("--analysis", choices=("bnb", "joint", "all"), default="all")
    inputs = tasks.add_parser("inputs", help="Visual input checks")
    inputs.add_argument("--kind", choices=("public", "numi-flux"), default="public")
    tasks.add_parser("compare", help="Render completed scan CSVs without fitting or recalibrating")
    miniboone = tasks.add_parser("miniboone", help="MiniBooNE 2020 combined official release or local likelihood reconstruction")
    miniboone.add_argument("--kind", choices=("official", "scan"), default="official")
    lsnd = tasks.add_parser("lsnd", help="LSND final-publication facts or 3+1 appearance-convention audit")
    lsnd.add_argument(
        "--kind", choices=("official", "core-mapping", "rate-scan"),
        default="official",
    )
    for command in (scan, spectrum, prepare, inputs, tasks.choices["compare"], miniboone, lsnd):
        command.add_argument("--batch", help="Output batch label; default: unique UTC timestamp")
    args, remaining = parser.parse_known_args(argv)
    from sterile_fit.output import begin_output_batch
    try:
        begin_output_batch(getattr(args, "batch", None))
    except ValueError as error:
        parser.error(str(error))
    if args.task == "scan":
        from sterile_fit.scan import scan_three_plus_one, scan_one_plus_three_plus_one
        if any(x == "--cls-calibration" or x.startswith("--cls-calibration=") for x in remaining):
            parser.error("Use --calibration, not a second --cls-calibration setting")
        if args.model == "1+3+1" and args.calibration == "adaptive-toy":
            parser.error("1+3+1 has no adaptive-toy implementation; choose analytic or toy")
        forwarded = []
        if args.preset:
            import yaml
            preset = yaml.safe_load((ROOT / "configs/runs" / (args.preset + ".yaml")).read_text(encoding="utf-8"))
            if preset["model"] != args.model:
                parser.error(f"Preset {args.preset} is for model {preset['model']}")
            forwarded.extend(preset["arguments"])
        forwarded.extend(["--cls-calibration", args.calibration, *remaining])
        if args.engine_help:
            forwarded = ["--help"]
        return _invoke(scan_three_plus_one if args.model == "3+1" else scan_one_plus_three_plus_one, forwarded)
    if args.task == "spectrum":
        from sterile_fit.experiments.microboone.bnb import plot_bnb_spectrum
        from sterile_fit.experiments.microboone.joint import plot_joint_spectrum, plot_figure1_spectrum
        return _invoke({"bnb": plot_bnb_spectrum, "joint": plot_joint_spectrum, "figure1": plot_figure1_spectrum}[args.kind], remaining)
    if args.task == "prepare":
        from sterile_fit.experiments.microboone.bnb import prepare_bnb_kernel, prepare_bnb_covariance
        from sterile_fit.experiments.microboone.numi import prepare_numi_flux, prepare_numi_kernel
        from sterile_fit.experiments.microboone.response import prepare_normalized_response, prepare_bnb26_response
        if args.kind.startswith("numi") and remaining:
            parser.error("NuMI preparation uses its declared constants and accepts no extra options")
        return _invoke({"reco-normalize": prepare_normalized_response, "reco-bnb26": prepare_bnb26_response,
                        "bnb-covariance": prepare_bnb_covariance, "bnb-kernel": prepare_bnb_kernel,
                        "numi-flux": prepare_numi_flux, "numi-kernel": prepare_numi_kernel}[args.kind], remaining)
    if args.task == "check":
        if remaining: parser.error("Unknown check options: " + " ".join(remaining))
        from sterile_fit.experiments.microboone.adapter import check_selected_inputs
        return check_selected_inputs(args.analysis)
    if args.task == "inputs":
        from sterile_fit.output import plot_public_inputs, plot_numi_flux_inputs
        if args.kind == "public" and remaining: parser.error("Public input plotting accepts no extra options")
        return _invoke(plot_public_inputs if args.kind == "public" else plot_numi_flux_inputs, remaining)
    if args.task == "compare":
        from sterile_fit.output import plot_completed_scan_contours
        return _invoke(plot_completed_scan_contours, remaining)
    if args.task == "miniboone":
        from sterile_fit.experiments.miniboone.adapter import run_miniboone
        return _invoke(run_miniboone, ["--kind", args.kind, *remaining])
    if args.task == "lsnd":
        from sterile_fit.experiments.lsnd.adapter import run_lsnd
        return _invoke(run_lsnd, ["--kind", args.kind, *remaining])


if __name__ == "__main__":
    main()

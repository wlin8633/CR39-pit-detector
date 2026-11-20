"""Reproduce and package the private CR-39 etching calibration benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cr39_detector.etching import EtchingTrackRatioFitter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=BENCHMARK_DIR / "local_paths.json",
        help="Private JSON configuration (default: local_paths.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report conventions without fitting.",
    )
    parser.add_argument(
        "--generate-curve",
        action="store_true",
        help="Generate the full forward curve after fitting.",
    )
    parser.add_argument(
        "--generate-lookup",
        action="store_true",
        help="Generate the forward curve and slow energy-angle lookup mesh.",
    )
    parser.add_argument(
        "--force-lookup",
        action="store_true",
        help="Generate a lookup mesh despite fit-quality warnings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the output NPZ path.",
    )
    parser.add_argument(
        "--from-fit",
        type=Path,
        help="Resume curve or mesh generation from an existing fit bundle.",
    )
    parser.add_argument(
        "--compare-only",
        nargs=2,
        metavar=("NEW_NPZ", "REFERENCE_NPZ"),
        help="Compare two existing bundles and exit without loading config.",
    )
    return parser.parse_args()


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {path}\n"
            "Copy local_paths.example.json to local_paths.json and edit it."
        )
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    for key in (
        "rel_npz",
        "range_npz",
        "experimental_pickle",
        "geant4_output_folder",
        "detection_results_folder",
        "detection_state_folder",
        "reference_npz",
        "output_directory",
    ):
        config[key] = resolve_path(config[key], path.parent)
    config["config_path"] = path
    return config


def load_inputs(config):
    for key in ("rel_npz", "range_npz", "experimental_pickle"):
        if not config[key].exists():
            raise FileNotFoundError(f"Required input does not exist: {config[key]}")

    with np.load(config["rel_npz"], allow_pickle=False) as data:
        rel_data = tuple(data[key].copy() for key in ("REL_x_E", "x", "E"))
    with np.load(config["range_npz"], allow_pickle=False) as data:
        range_data = tuple(data[key].copy() for key in ("R0_E", "E"))

    # Only open a pickle that comes from a trusted local experiment archive.
    with config["experimental_pickle"].open("rb") as stream:
        raw = pickle.load(stream)

    start = int(config["selection_start_index"])
    selection = slice(start, None)
    axis_multiplier = float(config["experimental_axis_multiplier_for_fit"])
    scale = float(config["experimental_axis_scale_to_um"]) * axis_multiplier
    experimental = {
        "E_exp": np.asarray(raw["energy"][selection], dtype=float),
        "a_exp": np.asarray(raw["a"][selection], dtype=float) * scale,
        "b_exp": np.asarray(raw["b"][selection], dtype=float) * scale,
    }
    for source_key, output_key in (("a_std", "a_std_exp"), ("b_std", "b_std_exp")):
        if source_key in raw:
            experimental[output_key] = (
                np.asarray(raw[source_key][selection], dtype=float) * scale
            )
    return rel_data, range_data, experimental


def compare_bundles(new_path, reference_path):
    new_path = Path(new_path).expanduser().resolve()
    reference_path = Path(reference_path).expanduser().resolve()
    print(f"New bundle:       {new_path}")
    print(f"Reference bundle: {reference_path}")
    with np.load(new_path, allow_pickle=False) as new, np.load(
        reference_path, allow_pickle=False
    ) as reference:
        common = sorted(set(new.files) & set(reference.files))
        for key in common:
            left = np.asarray(new[key])
            right = np.asarray(reference[key])
            if left.shape != right.shape or left.dtype.kind not in "iufc" or right.dtype.kind not in "iufc":
                continue
            finite = np.isfinite(left) & np.isfinite(right)
            max_abs = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else float("nan")
            print(f"{key:24s} shape={str(left.shape):12s} max_abs_delta={max_abs:.6g}")


def fit_quality_warnings(results):
    """Return strong fit/pathology warnings that should block a production mesh."""
    warnings = []
    finite_fraction = float(results.get("fit_finite_fraction", 1.0))
    if finite_fraction < 1.0:
        warnings.append(
            f"only {finite_fraction:.1%} of fit-point predictions are finite"
        )
    for axis in ("a", "b"):
        r2_key = f"{axis}_r2"
        if r2_key in results and float(results[r2_key]) < 0:
            warnings.append(f"{axis}-axis R2 is negative ({float(results[r2_key]):.4g})")
    if (
        "alpha_relative_uncertainty" in results
        and float(results["alpha_relative_uncertainty"]) > 1.0
    ):
        warnings.append(
            "alpha uncertainty exceeds the magnitude of alpha "
            f"({float(results['alpha_relative_uncertainty']):.1%})"
        )
    if (
        "max_abs_parameter_correlation" in results
        and float(results["max_abs_parameter_correlation"]) > 0.995
    ):
        warnings.append(
            "maximum absolute parameter correlation exceeds 0.995 "
            f"({float(results['max_abs_parameter_correlation']):.6f})"
        )
    return warnings


def print_summary(config, experimental):
    energy = experimental["E_exp"]
    print(f"Configuration: {config['config_path']}")
    print(f"REL input:     {config['rel_npz']}")
    print(f"Range input:   {config['range_npz']}")
    print(f"Pit table:     {config['experimental_pickle']}")
    print(f"Geant4 raw:    {config['geant4_output_folder']}")
    print(f"States:        {config['detection_state_folder']}")
    print(
        f"Selection:     data[{config['selection_start_index']}:] -> "
        f"{len(energy)} points, {energy.min():.6g} to {energy.max():.6g} MeV"
    )
    print(
        f"Etching:       VB={config['bulk_etch_rate_um_per_h']} um/h, "
        f"t={config['etch_time_h']} h, "
        f"removed depth={config['bulk_etch_rate_um_per_h'] * config['etch_time_h']} um per surface"
    )
    print(f"REL unit:      {config['rel_unit']}")
    print(f"Axis:          {config['experimental_axis_convention']}")
    print(
        f"Axis factor:   {config['experimental_axis_multiplier_for_fit']} "
        f"({config['experimental_axis_multiplier_note']})"
    )
    print(f"Theta:         {config['theta_reference']}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        # Windows may otherwise use cp1252 and fail on the omega character in
        # the tracked REL filename.
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    if args.compare_only:
        compare_bundles(*args.compare_only)
        return

    config = load_config(args.config)
    rel_data, range_data, experimental = load_inputs(config)
    print_summary(config, experimental)
    if args.dry_run:
        print("Dry run completed; no fit or file write was performed.")
        return

    removed_depth_um = (
        float(config["bulk_etch_rate_um_per_h"]) * float(config["etch_time_h"])
    )
    fitter = EtchingTrackRatioFitter(
        REL_data=rel_data,
        R0_data=range_data,
        exp_data=(
            experimental["E_exp"],
            experimental["a_exp"],
            experimental["b_exp"],
            removed_depth_um,
        ),
        rel_unit=config["rel_unit"],
    )
    fit_config = config["curve_fit"]
    curve_fit_params = {
        "p0": fit_config["p0"],
        "bounds": (fit_config["lower_bounds"], fit_config["upper_bounds"]),
        "maxfev": int(fit_config["maxfev"]),
    }
    if "x_scale" in fit_config:
        curve_fit_params["x_scale"] = fit_config["x_scale"]
    if "optimizer_verbose" in fit_config:
        curve_fit_params["verbose"] = int(fit_config["optimizer_verbose"])
    if args.from_fit:
        checkpoint = args.from_fit.expanduser().resolve()
        with np.load(checkpoint, allow_pickle=False) as saved:
            results = {key: saved[key].copy() for key in saved.files}
        fitter.provide_results(results)
        print(
            "Loaded fit checkpoint: "
            f"alpha={float(results['alpha_fit']):.9g}, "
            f"beta={float(results['beta_fit']):.9g}, "
            f"theta={np.degrees(float(results['theta_fit'])):.6g} deg"
        )
        if "combined_rmse_um" not in results:
            fitter.evaluate_fit_quality(results, verbose=True)
    else:
        results = fitter.fit_parameters(curve_fit_params, verbose=True)
    results.update(experimental)
    quality_warnings = fit_quality_warnings(results)
    if quality_warnings:
        print("Fit-quality warnings:")
        for warning in quality_warnings:
            print(f"  - {warning}")

    metadata = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "REL_unit": config["rel_unit"],
        "restriction_energy_eV": config["restriction_energy_eV"],
        "bulk_etch_rate_um_per_h": config["bulk_etch_rate_um_per_h"],
        "bulk_etch_rate_reported_um_per_h": config["bulk_etch_rate_reported_um_per_h"],
        "bulk_etch_rate_reported_uncertainty_um_per_h": config["bulk_etch_rate_reported_uncertainty_um_per_h"],
        "etch_time_h": config["etch_time_h"],
        "removed_depth_per_surface_um": removed_depth_um,
        "selection_start_index": config["selection_start_index"],
        "selection_rule": f"legacy data[{config['selection_start_index']}:]",
        "experimental_axis_source_unit": config["experimental_axis_source_unit"],
        "experimental_axis_convention": config["experimental_axis_convention"],
        "experimental_axis_multiplier_for_fit": config["experimental_axis_multiplier_for_fit"],
        "experimental_axis_multiplier_note": config["experimental_axis_multiplier_note"],
        "forward_axis_convention": "full major and minor diameters in um",
        "theta_reference": config["theta_reference"],
        "lookup_energy_range": config["lookup_energy_range"],
        "lookup_theta_range_rad": config["lookup_theta_range_rad"],
        "REL_input_sha256": file_sha256(config["rel_npz"]),
        "range_input_sha256": file_sha256(config["range_npz"]),
        "experimental_input_sha256": file_sha256(config["experimental_pickle"]),
        "fit_quality_warning_count": len(quality_warnings),
        "fit_quality_warnings": " | ".join(quality_warnings),
    }

    if args.output:
        output = args.output.expanduser().resolve()
    elif args.from_fit:
        output = args.from_fit.expanduser().resolve()
    else:
        config["output_directory"].mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = config["output_directory"] / f"{stamp}-etching_calibration.npz"

    # Save the successful fit before any optional slow generation stage.
    fitter.save_results(output, results, metadata=metadata, verbose=True)

    if args.generate_curve or args.generate_lookup:
        fitter.generate_forward_curve(results, verbose=True)
        fitter.save_results(output, results, metadata=metadata, verbose=True)

    if args.generate_lookup:
        if quality_warnings and not args.force_lookup:
            print(
                "Lookup mesh skipped because the fit has strong quality warnings. "
                "Inspect the saved fit/curve; use --force-lookup only for diagnostic work."
            )
        else:
            fitter.generate_lookup_mesh(
                results,
                energy_range=tuple(config["lookup_energy_range"]),
                theta_range=tuple(config["lookup_theta_range_rad"]),
                verbose=True,
            )
            fitter.save_results(output, results, metadata=metadata, verbose=True)

    if config["reference_npz"].exists():
        print("\nComparison with configured reference:")
        compare_bundles(output, config["reference_npz"])


if __name__ == "__main__":
    main()

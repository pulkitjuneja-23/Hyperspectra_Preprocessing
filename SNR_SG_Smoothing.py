"""
Compute Signal-to-Noise Ratio (SNR) per wavelength from a hyperspectral
reflectance CSV.

Input format:
- Columns: Study, Date, Plot, LAI, Growth_stage, plus ~270 wavelength columns
- Wavelength columns may be named like "398", "398nm", "398.5 nm", or "398_median"
- Rows: ~493 observations (plots)

Outputs:
- Prints summary stats for top noisy wavelengths
- Saves a CSV with per-wavelength metrics (mean, std, SNR, median, MAD, robust SNR)
- Optional plots: wavelength_nm vs CV; wavelength_nm vs SNR
- Interband correlation (adjacent bands) with low-correlation flags
- Optional: Savitzky-Golay smoothed data CSV with full analysis
 - Average reflectance vs wavelength (raw & smoothed) and first-derivative plots

Run:
    python SNR_SG_Smoothing.py --csv "path/to/ALLStudies_HSI_Median.csv" --out "SNR_by_wavelength.csv"
    python SNR_SG_Smoothing.py --smooth --plot  # Apply SG smoothing and analyze

Notes:
- SNR (mean/std) assumes variation across plots approximates noise; robust SNR
  uses median/MAD (scaled) to reduce sensitivity to outliers.
- This script only computes SNR; smoothing/derivatives can be applied later.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pandas is required to read CSVs. Install with: pip install pandas"
    ) from exc

try:
    from scipy import signal
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for Savitzky-Golay smoothing. Install with: pip install scipy"
    ) from exc

# Plotting is optional; only import when requested
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


META_COLUMNS = {"study", "date", "plot", "lai", "growth_stage"}
WAVELENGTH_MIN = 100.0  # nm, conservative lower bound to accept wider ranges
WAVELENGTH_MAX = 2500.0  # nm, conservative upper bound


def parse_wavelength(col: str) -> Optional[float]:
    """Extract numeric wavelength in nm from a column name.

    Accepts forms like "398", "398nm", "398.5 nm", "398_median", "  700 NM  ".
    Returns None for non-spectral columns.
    """
    if col.lower().strip() in META_COLUMNS:
        return None

    # Find first numeric token
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", col)
    if not m:
        return None
    try:
        wl = float(m.group(1))
    except ValueError:
        return None

    if not (WAVELENGTH_MIN <= wl <= WAVELENGTH_MAX):
        return None
    return wl


def detect_spectral_columns(df: "pd.DataFrame") -> List[Tuple[str, float]]:
    """Return list of (column_name, wavelength_nm) for spectral columns, sorted by nm."""
    spectral: List[Tuple[str, float]] = []
    for col in df.columns:
        wl = parse_wavelength(str(col))
        if wl is not None:
            spectral.append((col, wl))
    spectral.sort(key=lambda x: x[1])
    return spectral


def compute_snr(df: "pd.DataFrame", spectral_cols: List[Tuple[str, float]]) -> "pd.DataFrame":
    """Compute per-wavelength SNR using both mean/std and median/MAD (scaled)."""
    records: List[Dict[str, float]] = []
    for col, wl in spectral_cols:
        series = df[col].astype(float)
        count = series.count()
        mean = float(series.mean()) if count else float("nan")
        std = float(series.std(ddof=1)) if count > 1 else float("nan")
        snr_ms = mean / std if (std and std > 0) else float("inf") if mean != 0 else float("nan")

        med = float(series.median()) if count else float("nan")
        mad_raw = float((series - med).abs().median()) if count else float("nan")
        mad_scaled = mad_raw * 1.4826 if not math.isnan(mad_raw) else float("nan")
        snr_rm = med / mad_scaled if (mad_scaled and mad_scaled > 0) else float("inf") if med != 0 else float("nan")

        cv = std / mean if (std and mean and mean != 0) else float("nan")
        na_frac = float(series.isna().mean())

        records.append(
            {
                "wavelength_nm": wl,
                "column": col,
                "count": float(count),
                "mean": mean,
                "std": std,
                "snr_mean_std": snr_ms,
                "median": med,
                "mad_scaled": mad_scaled,
                "snr_median_mad": snr_rm,
                "cv": cv,
                "na_fraction": na_frac,
            }
        )

    import pandas as pd  # type: ignore

    result = pd.DataFrame.from_records(records)
    result.sort_values("wavelength_nm", inplace=True)
    return result


def compute_interband_correlation(df: "pd.DataFrame", spectral_cols: List[Tuple[str, float]]) -> "pd.DataFrame":
    """Compute correlation of each band with its immediate neighbors.

    corr_mean = average of corr(band, band-1) and corr(band, band+1) when available.
    Flags bands with corr_mean < 0.85.
    """
    import pandas as pd  # type: ignore

    records: List[Dict[str, float]] = []
    if len(spectral_cols) < 2:
        return pd.DataFrame(columns=["wavelength_nm", "column", "corr_prev", "corr_next", "corr_mean", "low_corr_flag"])

    # Pre-extract columns to avoid repeated lookups
    series_list = [(col, df[col].astype(float)) for col, _ in spectral_cols]

    for idx, (col, wl) in enumerate(spectral_cols):
        s = series_list[idx][1]
        corr_prev = float("nan")
        corr_next = float("nan")

        if idx > 0:
            s_prev = series_list[idx - 1][1]
            corr_prev = float(s.corr(s_prev)) if s.count() and s_prev.count() else float("nan")
        if idx < len(spectral_cols) - 1:
            s_next = series_list[idx + 1][1]
            corr_next = float(s.corr(s_next)) if s.count() and s_next.count() else float("nan")

        neighbors = [c for c in (corr_prev, corr_next) if not math.isnan(c)]
        corr_mean = float(np.mean(neighbors)) if neighbors else float("nan")
        low_corr_flag = bool(corr_mean < 0.85) if not math.isnan(corr_mean) else False

        records.append(
            {
                "wavelength_nm": wl,
                "column": col,
                "corr_prev": corr_prev,
                "corr_next": corr_next,
                "corr_mean": corr_mean,
                "low_corr_flag": low_corr_flag,
            }
        )

    result = pd.DataFrame.from_records(records)
    result.sort_values("wavelength_nm", inplace=True)
    return result


def apply_savgol_smoothing(
    df: "pd.DataFrame",
    spectral_cols: List[Tuple[str, float]],
    window_length: int = 9,
    polyorder: int = 2,
) -> "pd.DataFrame":
    """Apply Savitzky-Golay filter to all spectral columns, preserving metadata.

    Returns a new dataframe with smoothed spectral data and unchanged metadata columns.
    """
    import pandas as pd  # type: ignore

    smoothed_df = df.copy()
    spectral_col_names = [col for col, _ in spectral_cols]

    # Extract spectral matrix (n_samples, n_wavelengths)
    spectra = df[spectral_col_names].values.astype(float)

    # Apply SG filter along wavelength axis (axis=1)
    smoothed_spectra = signal.savgol_filter(spectra, window_length, polyorder, axis=1)

    # Replace columns
    smoothed_df[spectral_col_names] = smoothed_spectra
    return smoothed_df


def compute_first_derivative_matrix(
    df: "pd.DataFrame",
    spectral_cols: List[Tuple[str, float]],
) -> Tuple["pd.DataFrame", np.ndarray, np.ndarray]:
    """Compute first derivative dR/dλ along wavelength axis using np.gradient.

    Returns (derivative_df, wavelengths_nm, average_derivative_curve).
    """
    import pandas as pd  # type: ignore

    spectral_col_names = [col for col, _ in spectral_cols]
    wavelengths = np.array([wl for _, wl in spectral_cols], dtype=float)
    spectra = df[spectral_col_names].values.astype(float)

    deriv = np.gradient(spectra, wavelengths, axis=1)
    deriv_df = pd.DataFrame(deriv, columns=spectral_col_names, index=df.index)
    avg_deriv = deriv.mean(axis=0)
    return deriv_df, wavelengths, avg_deriv


def compute_average_curve(
    df: "pd.DataFrame",
    spectral_cols: List[Tuple[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average reflectance vs wavelength curve."""
    spectral_col_names = [col for col, _ in spectral_cols]
    wavelengths = np.array([wl for _, wl in spectral_cols], dtype=float)
    spectra = df[spectral_col_names].values.astype(float)
    avg_curve = spectra.mean(axis=0)
    return wavelengths, avg_curve


def save_csv_with_fallback(df: "pd.DataFrame", out_path: Path) -> Path:
    """Save CSV, falling back to an alternative filename if the target is locked.

    Useful when the default output is open in Excel/another app.
    """
    try:
        df.to_csv(out_path, index=False)
        return out_path
    except PermissionError:
        alt = out_path.with_name(out_path.stem + "_copy" + out_path.suffix)
        df.to_csv(alt, index=False)
        return alt


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute SNR per wavelength from hyperspectral CSV")
    ap.add_argument("--csv", default=None, help="Path to ALLStudies_HSI_Median.csv (default: same folder as script)")
    ap.add_argument("--out", default=None, help="Path to output SNR CSV (default: alongside input)")
    ap.add_argument(
        "--print-top",
        type=int,
        default=10,
        help="Print this many wavelengths with worst CV (noise)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Generate wavelength vs CV and SNR plots (requires matplotlib)",
    )
    ap.add_argument(
        "--plot-cv-path",
        default=None,
        help="Output path for CV plot PNG (default: alongside SNR CSV)",
    )
    ap.add_argument(
        "--plot-snr-path",
        default=None,
        help="Output path for SNR plot PNG (default: alongside SNR CSV)",
    )
    ap.add_argument(
        "--smooth",
        action="store_true",
        help="Apply Savitzky-Golay smoothing and save smoothed CSV with full analysis",
    )
    ap.add_argument(
        "--sg-window",
        type=int,
        default=9,
        help="Savitzky-Golay window length (must be odd, default: 9)",
    )
    ap.add_argument(
        "--sg-polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order (default: 2)",
    )
    args = ap.parse_args()

    # Determine CSV path: use provided path, otherwise look in script directory
    if args.csv:
        csv_path = Path(args.csv)
    else:
        script_dir = Path(__file__).resolve().parent
        default_csv = script_dir / "ALLStudies_HSI_Median.csv"
        if default_csv.exists():
            csv_path = default_csv
        else:
            # Fallback: first CSV found in directory
            candidates = list(script_dir.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError(
                    "No CSV provided and none found in script directory. "
                    "Place ALLStudies_HSI_Median.csv alongside this script or pass --csv."
                )
            csv_path = candidates[0]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    spectral_cols = detect_spectral_columns(df)
    if not spectral_cols:
        raise RuntimeError("No spectral columns detected. Ensure wavelength columns contain a numeric wavelength like '398' or '398_median'.")

    snr_df = compute_snr(df, spectral_cols)
    corr_df = compute_interband_correlation(df, spectral_cols)
    # Merge correlation results into snr_df for a single CSV output
    snr_df = snr_df.merge(corr_df, on=["wavelength_nm", "column"], how="left")

    # Show top noisy wavelengths by coefficient of variation (higher CV = more noise)
    noisy = snr_df.copy()
    noisy = noisy[~noisy["cv"].isna()]
    noisy.sort_values("cv", ascending=False, inplace=True)
    top_n = noisy.head(args.print_top)
    print("Top noisy wavelengths by CV:")
    for _, row in top_n.iterrows():
        print(
            f" {row['wavelength_nm']:.2f} nm (col '{row['column']}'): CV={row['cv']:.4f}, "
            f"SNR(mean/std)={row['snr_mean_std']:.2f}, SNR(median/MAD)={row['snr_median_mad']:.2f}"
        )

    # Save output
    out_path = Path(args.out) if args.out else csv_path.with_name("SNR_by_wavelength.csv")
    out_path = save_csv_with_fallback(snr_df, out_path)
    print(f"Saved SNR metrics to: {out_path}")

    # Save interband correlation as a dedicated file too
    corr_out = out_path.with_name("Interband_correlation.csv")
    corr_out = save_csv_with_fallback(corr_df, corr_out)
    print(f"Saved interband correlation to: {corr_out}")

    if args.plot:
        if plt is None:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )
        plot_cv_path = Path(args.plot_cv_path) if args.plot_cv_path else out_path.with_name("SNR_CV.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(snr_df["wavelength_nm"], snr_df["cv"], marker="o", linestyle="-", markersize=2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Coefficient of Variation (CV)")
        ax.set_title("Wavelength vs CV")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_cv_path, dpi=200)
        plt.close(fig)
        print(f"Saved CV plot to: {plot_cv_path}")

        plot_snr_path = Path(args.plot_snr_path) if args.plot_snr_path else out_path.with_name("SNR_plot.png")
        # Use snr_mean_std; drop non-finite for plotting
        snr_plot = snr_df[np.isfinite(snr_df["snr_mean_std"])]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(snr_plot["wavelength_nm"], snr_plot["snr_mean_std"], marker="o", linestyle="-", markersize=2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("SNR (mean/std)")
        ax.set_title("Wavelength vs SNR")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_snr_path, dpi=200)
        plt.close(fig)
        print(f"Saved SNR plot to: {plot_snr_path}")

        # Defer combined average plot until smoothing section (when available)

    # Apply Savitzky-Golay smoothing if requested
    if args.smooth:
        print(f"\nApplying Savitzky-Golay smoothing (window={args.sg_window}, polyorder={args.sg_polyorder})...")
        smoothed_df = apply_savgol_smoothing(df, spectral_cols, args.sg_window, args.sg_polyorder)

        # Save smoothed data
        smoothed_csv_path = csv_path.with_name("SG_smoothed_median_HSI.csv")
        smoothed_csv_path = save_csv_with_fallback(smoothed_df, smoothed_csv_path)
        print(f"Saved smoothed data to: {smoothed_csv_path}")

        # Re-run full analysis on smoothed data
        print("\nAnalyzing smoothed data...")
        snr_smoothed = compute_snr(smoothed_df, spectral_cols)
        corr_smoothed = compute_interband_correlation(smoothed_df, spectral_cols)
        snr_smoothed = snr_smoothed.merge(corr_smoothed, on=["wavelength_nm", "column"], how="left")

        # Save smoothed analysis
        snr_smoothed_path = csv_path.with_name("SNR_smoothed_by_wavelength.csv")
        snr_smoothed_path = save_csv_with_fallback(snr_smoothed, snr_smoothed_path)
        print(f"Saved smoothed SNR metrics to: {snr_smoothed_path}")

        corr_smoothed_path = csv_path.with_name("Interband_correlation_smoothed.csv")
        corr_smoothed_path = save_csv_with_fallback(corr_smoothed, corr_smoothed_path)
        print(f"Saved smoothed interband correlation to: {corr_smoothed_path}")

        # Generate plots for smoothed data if plotting enabled
        if args.plot:
            plot_cv_smoothed_path = csv_path.with_name("SNR_CV_smoothed.png")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(snr_smoothed["wavelength_nm"], snr_smoothed["cv"], marker="o", linestyle="-", markersize=2)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Coefficient of Variation (CV)")
            ax.set_title("Wavelength vs CV (Smoothed Data)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_cv_smoothed_path, dpi=200)
            plt.close(fig)
            print(f"Saved smoothed CV plot to: {plot_cv_smoothed_path}")

            plot_snr_smoothed_path = csv_path.with_name("SNR_plot_smoothed.png")
            snr_smoothed_plot = snr_smoothed[np.isfinite(snr_smoothed["snr_mean_std"])]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(snr_smoothed_plot["wavelength_nm"], snr_smoothed_plot["snr_mean_std"], marker="o", linestyle="-", markersize=2)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("SNR (mean/std)")
            ax.set_title("Wavelength vs SNR (Smoothed Data)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_snr_smoothed_path, dpi=200)
            plt.close(fig)
            print(f"Saved smoothed SNR plot to: {plot_snr_smoothed_path}")

            # Combined average plot: raw reflectance, smoothed reflectance, and first derivative
            wl_raw, avg_raw = compute_average_curve(df, spectral_cols)
            wl_sm, avg_sm = compute_average_curve(smoothed_df, spectral_cols)
            deriv_sm_df, wl_deriv, avg_deriv = compute_first_derivative_matrix(smoothed_df, spectral_cols)
            deriv_sm_csv = csv_path.with_name("First_derivative_smoothed.csv")
            deriv_sm_csv = save_csv_with_fallback(deriv_sm_df, deriv_sm_csv)
            print(f"Saved first derivative (smoothed) to: {deriv_sm_csv}")

            combined_path = csv_path.with_name("Average_Raw_Smoothed_Derivative.png")
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(wl_raw, avg_raw, linestyle="-", color="tab:blue", label="Raw Avg Reflectance")
            ax.plot(wl_sm, avg_sm, linestyle="-", color="tab:green", label="Smoothed Avg Reflectance")
            ax.plot(wl_deriv, avg_deriv, linestyle="-", color="tab:red", label="Avg First Derivative")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Value")
            ax.set_title("Average Reflectance and First Derivative vs Wavelength")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(combined_path, dpi=200)
            plt.close(fig)
            print(f"Saved combined averages plot to: {combined_path}")


if __name__ == "__main__":
    main()
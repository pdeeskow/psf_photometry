#!/usr/bin/env python
"""PSF photometry pipeline: solve WCS, match APASS, write CSV."""

import argparse
import math
import os
import sys
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import custom_model

from astroquery.astrometry_net import AstrometryNet
from astroquery.vizier import Vizier

from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry

try:
    from photutils.psf import CircularGaussianSigmaPRF
except ImportError:
    from photutils.psf import IntegratedGaussianPRF as CircularGaussianSigmaPRF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a FITS image, run PSF photometry, and match APASS V magnitudes."
    )
    parser.add_argument("--fits", required=True, help="Path to input FITS image")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument(
        "--plot",
        default=None,
        help="Path to output PNG plot (v_mag vs flux). Defaults to <out>_vmag_vs_flux.png",
    )
    parser.add_argument(
        "--api-key-env",
        default="ASTROMETRY_API_KEY",
        help="Environment variable with astrometry.net API key",
    )
    parser.add_argument("--fwhm", type=float, default=3.0, help="FWHM in pixels")
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=5.0,
        help="Detection threshold in sigma",
    )
    parser.add_argument(
        "--match-arcsec",
        type=float,
        default=2.0,
        help="Match radius in arcsec for APASS",
    )
    parser.add_argument(
        "--max-stars",
        type=int,
        default=500,
        help="Max sources to use for photometry",
    )
    parser.add_argument(
        "--apass-catalog",
        default="II/336/apass9",
        help="VizieR catalog name for APASS",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Astrometry.net poll interval in seconds",
    )
    parser.add_argument(
        "--solve-timeout",
        type=float,
        default=900.0,
        help="Astrometry.net solve timeout in seconds",
    )
    parser.add_argument(
        "--fit-seed-density",
        type=int,
        default=1,
        help="Seed density for multi-start logistic fit (>=1)",
    )
    return parser.parse_args()


def load_image(path: str) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = np.array(hdu.data, dtype=float)
                header = hdu.header.copy()
                break
        else:
            raise ValueError("No image data found in FITS")
    if data.ndim != 2:
        raise ValueError("Only 2D FITS images are supported")
    return data, header


def solve_wcs(path: str, api_key: str, poll_interval: float, timeout: float) -> fits.Header:
    ast = AstrometryNet()
    ast.api_key = api_key
    
    try:
        wcs_header = ast.solve_from_image(path)
        if wcs_header is None:
            raise RuntimeError("Astrometry.net solve failed")
    except Exception as e:
        raise RuntimeError(f"Astrometry.net solve error: {e}")
    return wcs_header


def compute_search_radius(wcs: WCS, shape: tuple[int, int]) -> u.Quantity:
    ny, nx = shape
    corners = np.array(
        [[0, 0], [0, ny - 1], [nx - 1, 0], [nx - 1, ny - 1]], dtype=float
    )
    world = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
    center = wcs.pixel_to_world(nx / 2.0, ny / 2.0)
    separations = center.separation(world)
    return separations.max()


def query_apass(center: SkyCoord, radius: u.Quantity, catalog: str) -> Table:
    vizier = Vizier(columns=["RAJ2000", "DEJ2000", "Vmag", "B-V", "Bmag", "e_Vmag", "ID"])
    vizier.ROW_LIMIT = -1
    result = vizier.query_region(center, radius=radius, catalog=catalog)
    if not result:
        return Table()
    return result[0]


def detect_sources(data: np.ndarray, fwhm: float, threshold_sigma: float, max_stars: int) -> Table:
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = finder(data - median)
    if sources is None:
        return Table()
    if len(sources) > max_stars:
        sources = sources[:max_stars]
    return sources


def run_psf_photometry(
    data: np.ndarray, sources: Table, fwhm: float
) -> Table:
    if len(sources) == 0:
        return Table()

    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    psf_model = CircularGaussianSigmaPRF(sigma=sigma)
    fitter = LevMarLSQFitter()

    phot = PSFPhotometry(
        psf_model=psf_model,
        fitter=fitter,
        fit_shape=(11, 11),
        aperture_radius=1.5 * fwhm,
    )
    init_params = Table()
    init_params["x_0"] = sources["xcentroid"]
    init_params["y_0"] = sources["ycentroid"]
    if "flux" in sources.colnames:
        init_params["flux_0"] = sources["flux"]
    return phot(data, init_params=init_params)


def resolve_flux_column(phot_table: Table) -> str:
    for name in ("flux_fit", "flux_0", "flux"):
        if name in phot_table.colnames:
            return name
    raise KeyError("No flux column found in photometry output")


def resolve_apass_column(apass: Table, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in apass.colnames:
            return name
    raise KeyError(f"No {label} column found in APASS output. Columns: {apass.colnames}")


def _logistic_flux_eval(mag: np.ndarray, flux_low: float, flux_high: float, m0: float, slope: float) -> np.ndarray:
    z = np.clip(slope * (mag - m0), -700.0, 700.0)
    return flux_low + (flux_high - flux_low) / (1.0 + np.exp(z))


def _invert_flux_to_vmag_numeric(
    flux_values: np.ndarray,
    flux_low: float,
    flux_high: float,
    m0: float,
    slope: float,
    mag_lo: float,
    mag_hi: float,
    max_iter: int = 80,
) -> np.ndarray:
    flux_arr = np.asarray(flux_values, dtype=float)
    out = np.full_like(flux_arr, np.nan, dtype=float)

    if slope <= 0.0 or flux_high <= flux_low:
        return out

    f_lo = _logistic_flux_eval(np.array([mag_lo]), flux_low, flux_high, m0, slope)[0]
    f_hi = _logistic_flux_eval(np.array([mag_hi]), flux_low, flux_high, m0, slope)[0]
    f_min = min(f_lo, f_hi)
    f_max = max(f_lo, f_hi)

    valid = np.isfinite(flux_arr) & (flux_arr > f_min) & (flux_arr < f_max)
    if not np.any(valid):
        return out

    target = flux_arr[valid]
    lo = np.full(target.shape, mag_lo, dtype=float)
    hi = np.full(target.shape, mag_hi, dtype=float)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _logistic_flux_eval(mid, flux_low, flux_high, m0, slope)
        move_right = f_mid > target
        lo[move_right] = mid[move_right]
        hi[~move_right] = mid[~move_right]

    out[np.where(valid)[0]] = 0.5 * (lo + hi)
    return out


def _fit_vmag_calibration(flux: np.ndarray, vmag: np.ndarray, seed_density: int = 1):
    fit_mask = np.isfinite(flux) & (flux > 0.0) & np.isfinite(vmag)
    if np.count_nonzero(fit_mask) < 2:
        return "none", None, {}

    x_mag = vmag[fit_mask]
    y_flux = flux[fit_mask]

    if np.count_nonzero(fit_mask) >= 5:
        mag_min = float(np.nanmin(x_mag))
        mag_max = float(np.nanmax(x_mag))

        bright_cut = float(np.nanpercentile(x_mag, 15))
        faint_cut = float(np.nanpercentile(x_mag, 85))
        bright_mask = x_mag <= bright_cut
        faint_mask = x_mag >= faint_cut

        if np.any(bright_mask):
            flux_high_seed = float(np.nanmedian(y_flux[bright_mask]))
        else:
            flux_high_seed = float(np.nanpercentile(y_flux, 90))

        if np.any(faint_mask):
            flux_low_seed = float(np.nanmedian(y_flux[faint_mask]))
        else:
            flux_low_seed = float(np.nanpercentile(y_flux, 10))

        if flux_high_seed <= flux_low_seed:
            flux_low_seed = float(np.nanpercentile(y_flux, 10))
            flux_high_seed = float(np.nanpercentile(y_flux, 90))

        y_min_obs = float(np.nanmin(y_flux))
        y_max_obs = float(np.nanmax(y_flux))
        flux_low_seed = max(0.0, min(flux_low_seed, 0.99 * y_min_obs))
        flux_high_seed = max(flux_high_seed, 1.01 * y_max_obs)

        seed_density = max(1, int(seed_density))
        mag_span = max(0.1, float(np.nanpercentile(x_mag, 90) - np.nanpercentile(x_mag, 10)))
        q_count = 3 + 2 * seed_density
        quantiles = np.linspace(20.0, 80.0, q_count)
        m0_seeds = np.nanpercentile(x_mag, quantiles)
        slope_base = max(0.08, 4.0 / mag_span)
        slope_factors = np.linspace(0.5, 1.8, 2 + seed_density)
        slope_seeds = [min(3.5, max(0.05, float(slope_base * factor))) for factor in slope_factors]

        fitter = LevMarLSQFitter()
        best_model = None
        best_score = np.inf

        for m0_seed in m0_seeds:
            for slope_seed in slope_seeds:
                model = logistic_flux_from_vmag(
                    flux_low=flux_low_seed,
                    flux_high=flux_high_seed,
                    m0=float(m0_seed),
                    slope=float(slope_seed),
                )
                model.flux_low.bounds = (0.0, max(1.0, 1.02 * y_min_obs))
                model.flux_high.bounds = (max(1.01 * y_max_obs, flux_low_seed + 1.0), y_max_obs * 4.0)
                model.m0.bounds = (mag_min - 2.0, mag_max + 2.0)
                model.slope.bounds = (0.03, 4.0)

                try:
                    fitted_model = fitter(model, x_mag, y_flux)
                except Exception:
                    continue

                pred = fitted_model(x_mag)
                if not np.all(np.isfinite(pred)):
                    continue
                score = float(np.mean((pred - y_flux) ** 2))
                if score < best_score:
                    best_score = score
                    best_model = fitted_model

        if best_model is not None:
            flux_low = float(best_model.flux_low.value)
            flux_high = float(best_model.flux_high.value)
            m0 = float(best_model.m0.value)
            slope = float(best_model.slope.value)

            if slope > 0.0 and flux_high > flux_low:
                def inverse_model(flux_values: np.ndarray) -> np.ndarray:
                    return _invert_flux_to_vmag_numeric(
                        flux_values=np.asarray(flux_values, dtype=float),
                        flux_low=flux_low,
                        flux_high=flux_high,
                        m0=m0,
                        slope=slope,
                        mag_lo=mag_min - 8.0,
                        mag_hi=mag_max + 8.0,
                    )

                fit_params = {
                    "fit_model": "inverse_logistic_from_flux_numeric",
                    "fit_flux_low": flux_low,
                    "fit_flux_high": flux_high,
                    "fit_m0": m0,
                    "fit_slope": slope,
                    "fit_seed_density": float(seed_density),
                    "fit_n_refstars": float(np.count_nonzero(fit_mask)),
                }
                return "inverse_logistic_from_flux_numeric", inverse_model, fit_params

    try:
        slope, intercept = np.polyfit(np.log10(y_flux), x_mag, 1)

        def linear_model(flux_values: np.ndarray) -> np.ndarray:
            return slope * np.log10(flux_values) + intercept

        fit_params = {
            "fit_model": "linear_log_flux_fallback",
            "fit_slope": float(slope),
            "fit_linear_intercept": float(intercept),
            "fit_seed_density": float(max(1, int(seed_density))),
            "fit_n_refstars": float(np.count_nonzero(fit_mask)),
        }
        return "linear_log_flux_fallback", linear_model, fit_params
    except Exception:
        return "none", None, {}


def plot_vmag_vs_flux(output: Table, plot_path: str, seed_density: int = 1) -> None:
    import matplotlib.pyplot as plt

    flux_values = []
    vmag_values = []
    bv_values = []

    for flux, vmag, bv in zip(output["flux"], output["v_mag"], output["b-v"]):
        try:
            flux_val = float(flux)
            vmag_val = float(vmag)
            bv_val = float(bv)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(flux_val) or not np.isfinite(vmag_val):
            continue
        if not np.isfinite(bv_val):
            continue
        if flux_val <= 0:
            continue

        flux_values.append(flux_val)
        vmag_values.append(vmag_val)
        bv_values.append(bv_val)

    flux_fit_values = []
    vmag_fit_values = []
    for flux, vmag in zip(output["flux"], output["v_mag"]):
        try:
            flux_val = float(flux)
            vmag_val = float(vmag)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(flux_val) or not np.isfinite(vmag_val) or flux_val <= 0:
            continue
        flux_fit_values.append(flux_val)
        vmag_fit_values.append(vmag_val)

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(flux_values) > 0:
        scatter = ax.scatter(
            flux_values,
            vmag_values,
            c=bv_values,
            cmap="RdBu",
            s=14,
            alpha=0.8,
        )
        ax.set_xscale("log")
        ax.set_xlabel("flux")
        ax.set_ylabel("v_mag")
        ax.set_title("APASS v_mag vs PSF flux (color: b-v)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        if len(flux_fit_values) >= 2:
            fit_flux = np.asarray(flux_fit_values, dtype=float)
            fit_vmag = np.asarray(vmag_fit_values, dtype=float)
            fit_method, fit_model, _ = _fit_vmag_calibration(fit_flux, fit_vmag, seed_density=seed_density)
            if fit_model is not None:
                log_x = np.linspace(np.log10(np.min(fit_flux)), np.log10(np.max(fit_flux)), 256)
                curve_flux = 10 ** log_x
                curve_vmag = fit_model(curve_flux)
                valid_curve = np.isfinite(curve_vmag)
                if np.any(valid_curve):
                    ax.plot(curve_flux[valid_curve], curve_vmag[valid_curve], color="black", lw=1.5, label=f"Regression ({fit_method})")
                    ax.legend(loc="best")

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("b-v")
    else:
        ax.text(0.5, 0.5, "No matched APASS V magnitudes with b-v", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_vmagcal_vs_vmag(output: Table, plot_path: str) -> None:
    import matplotlib.pyplot as plt

    vmag_ref = []
    vmag_cal = []
    bv_values = []

    for vmag, vmagc, bv in zip(output["v_mag"], output["v_mag_cal"], output["b-v"]):
        try:
            vmag_val = float(vmag)
            vmagc_val = float(vmagc)
            bv_val = float(bv)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(vmag_val) or not np.isfinite(vmagc_val) or not np.isfinite(bv_val):
            continue

        vmag_ref.append(vmag_val)
        vmag_cal.append(vmagc_val)
        bv_values.append(bv_val)

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(vmag_ref) > 0:
        vmag_ref_arr = np.asarray(vmag_ref, dtype=float)
        vmag_cal_arr = np.asarray(vmag_cal, dtype=float)
        delta = vmag_cal_arr - vmag_ref_arr
        mean_error = float(np.mean(delta))
        mean_abs_error = float(np.mean(np.abs(delta)))
        rmse = float(np.sqrt(np.mean(delta ** 2)))

        scatter = ax.scatter(
            vmag_ref,
            vmag_cal,
            c=bv_values,
            cmap="RdBu",
            s=16,
            alpha=0.85,
        )

        low = min(min(vmag_ref), min(vmag_cal))
        high = max(max(vmag_ref), max(vmag_cal))
        pad = 0.2 * (high - low) if high > low else 0.5
        line_min = low - pad
        line_max = high + pad
        ax.plot([line_min, line_max], [line_min, line_max], "k--", lw=1.2, label="v_mag = v_mag_cal")

        ax.set_xlim(line_min, line_max)
        ax.set_ylim(line_min, line_max)
        ax.set_xlabel("v_mag (APASS)")
        ax.set_ylabel("v_mag_cal (calibrated)")
        ax.set_title("Calibration QC: v_mag_cal vs v_mag")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.text(
            0.02,
            0.02,
            (
                f"mean error (cal-ref): {mean_error:+.4f} mag\n"
                f"mean abs error: {mean_abs_error:.4f} mag\n"
                f"RMSE: {rmse:.4f} mag"
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("b-v")
    else:
        ax.text(0.5, 0.5, "No stars with valid v_mag, v_mag_cal and b-v", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


@custom_model
def logistic_flux_from_vmag(mag, flux_low=1000.0, flux_high=1e6, m0=13.0, slope=1.0):
    return flux_low + (flux_high - flux_low) / (1.0 + np.exp(slope * (mag - m0)))


def _to_float_array(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    for i, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out[i] = number
    return out


def add_calibrated_vmag(output: Table, seed_density: int = 1) -> tuple[str, dict[str, float | str]]:
    if len(output) == 0:
        output["v_mag_cal"] = np.array([], dtype=float)
        output["fit_model"] = np.array([], dtype=object)
        output["fit_flux_low"] = np.array([], dtype=float)
        output["fit_flux_high"] = np.array([], dtype=float)
        output["fit_m0"] = np.array([], dtype=float)
        output["fit_slope"] = np.array([], dtype=float)
        output["fit_linear_intercept"] = np.array([], dtype=float)
        output["fit_seed_density"] = np.array([], dtype=float)
        output["fit_n_refstars"] = np.array([], dtype=float)
        return "none", {}

    flux_arr = _to_float_array(np.asarray(output["flux"]))
    vmag_arr = _to_float_array(np.asarray(output["v_mag"]))

    fit_mask = np.isfinite(flux_arr) & (flux_arr > 0.0) & np.isfinite(vmag_arr)
    pred_mask = np.isfinite(flux_arr) & (flux_arr > 0.0)

    vmag_cal = np.full(len(output), np.nan, dtype=float)

    fit_method, fit_model, fit_params = _fit_vmag_calibration(
        flux_arr[fit_mask],
        vmag_arr[fit_mask],
        seed_density=seed_density,
    )
    if fit_model is not None and pred_mask.any():
        vmag_cal[pred_mask] = fit_model(flux_arr[pred_mask])

    output["v_mag_cal"] = vmag_cal
    output["fit_model"] = np.full(len(output), fit_method, dtype=object)

    for column in ("fit_flux_low", "fit_flux_high", "fit_m0", "fit_slope", "fit_linear_intercept", "fit_seed_density", "fit_n_refstars"):
        value = fit_params.get(column, np.nan)
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            scalar = np.nan
        output[column] = np.full(len(output), scalar, dtype=float)

    return fit_method, fit_params


def build_output(
    phot_table: Table,
    wcs: WCS,
    apass: Table,
    match_radius: u.Quantity,
) -> Table:
    if len(phot_table) == 0:
        return Table(
            names=(
                "RA",
                "dec",
                "flux",
                "v_mag",
                "b-v",
                "v_mag_cal",
                "fit_model",
                "fit_flux_low",
                "fit_flux_high",
                "fit_m0",
                "fit_slope",
                "fit_linear_intercept",
                "fit_seed_density",
                "fit_n_refstars",
            )
        )

    x_col = "x_fit" if "x_fit" in phot_table.colnames else "x_0"
    y_col = "y_fit" if "y_fit" in phot_table.colnames else "y_0"
    sky = wcs.pixel_to_world(phot_table[x_col], phot_table[y_col])

    flux_col = resolve_flux_column(phot_table)
    out = Table()
    out["RA"] = sky.ra.deg
    out["dec"] = sky.dec.deg
    out["flux"] = phot_table[flux_col]
    out["v_mag"] = np.full(len(phot_table), "", dtype=object)
    out["b-v"] = np.full(len(phot_table), "", dtype=object)

    if len(apass) > 0:
        ra_col = resolve_apass_column(apass, ("RAJ2000", "RA_ICRS", "RAdeg", "RA"), "RA")
        dec_col = resolve_apass_column(apass, ("DEJ2000", "DE_ICRS", "DEdeg", "DEC", "Dec"), "DEC")
        vmag_col = resolve_apass_column(apass, ("Vmag", "V", "Vmag1"), "V magnitude")

        bv_col = None
        for candidate in ("B-V", "B_V", "BV", "b-v"):
            if candidate in apass.colnames:
                bv_col = candidate
                break

        use_computed_bv = False
        bmag_col = None
        if bv_col is None and "Bmag" in apass.colnames:
            use_computed_bv = True
            bmag_col = "Bmag"

        apass_coords = SkyCoord(apass[ra_col], apass[dec_col], unit=(u.deg, u.deg))
        idx, sep2d, _ = sky.match_to_catalog_sky(apass_coords)
        matched = sep2d <= match_radius

        matched_indices = np.where(matched)[0]
        out["v_mag"][matched_indices] = np.asarray(apass[vmag_col][idx][matched], dtype=str)
        if bv_col is not None:
            out["b-v"][matched_indices] = np.asarray(apass[bv_col][idx][matched], dtype=str)
        elif use_computed_bv and bmag_col is not None:
            bmag_values = np.asarray(apass[bmag_col][idx][matched], dtype=float)
            vmag_values = np.asarray(apass[vmag_col][idx][matched], dtype=float)
            out["b-v"][matched_indices] = np.asarray(bmag_values - vmag_values, dtype=str)

    return out


def main() -> int:
    args = parse_args()
    args.fit_seed_density = max(1, int(args.fit_seed_density))

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Missing astrometry.net API key in {args.api_key_env}")
        return 2

    data, header = load_image(args.fits)
    wcs_header = solve_wcs(args.fits, api_key, args.poll_interval, args.solve_timeout)
    header.update(wcs_header)
    wcs = WCS(header)

    radius = compute_search_radius(wcs, data.shape)
    center = wcs.pixel_to_world(data.shape[1] / 2.0, data.shape[0] / 2.0)

    apass = query_apass(center, radius, args.apass_catalog)
    sources = detect_sources(data, args.fwhm, args.threshold_sigma, args.max_stars)
    phot_table = run_psf_photometry(data, sources, args.fwhm)

    output = build_output(phot_table, wcs, apass, args.match_arcsec * u.arcsec)
    fit_method, fit_params = add_calibrated_vmag(output, seed_density=args.fit_seed_density)
    output.write(args.out, format="csv", overwrite=True)

    plot_path = args.plot if args.plot else f"{os.path.splitext(args.out)[0]}_vmag_vs_flux.png"
    plot_vmag_vs_flux(output, plot_path, seed_density=args.fit_seed_density)
    qc_plot_path = f"{os.path.splitext(args.out)[0]}_vmagcal_vs_vmag.png"
    plot_vmagcal_vs_vmag(output, qc_plot_path)

    print(f"Wrote {len(output)} rows to {args.out}")
    print(f"v_mag calibration model: {fit_method}")
    if fit_params:
        if fit_method == "inverse_logistic_from_flux_numeric":
            print(
                "Fit params: "
                f"flux_low={fit_params.get('fit_flux_low', np.nan):.3f}, "
                f"flux_high={fit_params.get('fit_flux_high', np.nan):.3f}, "
                f"m0={fit_params.get('fit_m0', np.nan):.4f}, "
                f"slope={fit_params.get('fit_slope', np.nan):.4f}, "
                f"seed_density={int(fit_params.get('fit_seed_density', args.fit_seed_density))}, "
                f"n_refstars={int(fit_params.get('fit_n_refstars', 0))}"
            )
        elif fit_method == "linear_log_flux_fallback":
            print(
                "Fallback params: "
                f"slope={fit_params.get('fit_slope', np.nan):.6f}, "
                f"intercept={fit_params.get('fit_linear_intercept', np.nan):.6f}, "
                f"seed_density={int(fit_params.get('fit_seed_density', args.fit_seed_density))}, "
                f"n_refstars={int(fit_params.get('fit_n_refstars', 0))}"
            )
    print(f"Wrote plot to {plot_path}")
    print(f"Wrote calibration QC plot to {qc_plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

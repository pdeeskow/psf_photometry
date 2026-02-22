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


def plot_vmag_vs_flux(output: Table, plot_path: str) -> None:
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
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("b-v")
    else:
        ax.text(0.5, 0.5, "No matched APASS V magnitudes with b-v", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def build_output(
    phot_table: Table,
    wcs: WCS,
    apass: Table,
    match_radius: u.Quantity,
) -> Table:
    if len(phot_table) == 0:
        return Table(names=("RA", "dec", "flux", "v_mag", "b-v"))

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
    output.write(args.out, format="csv", overwrite=True)

    plot_path = args.plot if args.plot else f"{os.path.splitext(args.out)[0]}_vmag_vs_flux.png"
    plot_vmag_vs_flux(output, plot_path)

    print(f"Wrote {len(output)} rows to {args.out}")
    print(f"Wrote plot to {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""PSF photometry pipeline: solve WCS, match APASS, write CSV."""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import custom_model
from astropy.utils.exceptions import AstropyDeprecationWarning
from scipy.optimize import curve_fit as _scipy_curve_fit

from astroquery.astrometry_net import AstrometryNet
from astroquery.vizier import Vizier

from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry

# ── Suppress noisy but harmless warnings ──────────────────────────────────────
# FITS cards returned by astrometry.net are often non-standard
warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
# datfix / WCS-axis bookkeeping fixes applied automatically by astropy
warnings.filterwarnings("ignore", message=".*datfix.*", category=fits.verify.VerifyWarning)
warnings.filterwarnings("ignore", message=".*datfix.*")
warnings.filterwarnings("ignore", message=".*more axes.*than the image.*")
# astroquery deprecation on force_image_upload (already handled in code)
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning,
                        message=".*force_image_upload.*")

try:
    from photutils.psf import CircularGaussianSigmaPRF
except ImportError:
    from photutils.psf import IntegratedGaussianPRF as CircularGaussianSigmaPRF


# ── Band configuration ───────────────────────────────────────────────────────
_BAND_CONFIG: dict[str, dict] = {
    "V": {
        "apass_mag_candidates": ("Vmag", "V", "Vmag1"),
        "out_col": "v_mag",
        "cal_col": "v_mag_cal",
        "label": "V (Johnson)",
    },
    "B": {
        "apass_mag_candidates": ("Bmag", "B"),
        "out_col": "b_mag",
        "cal_col": "b_mag_cal",
        "label": "B (Johnson)",
    },
    "R": {
        "apass_mag_candidates": ("r'mag", "Rmag", "r_mag"),
        "out_col": "r_mag",
        "cal_col": "r_mag_cal",
        "label": "r' (Sloan)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a FITS image, run PSF photometry, and match APASS V magnitudes."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fits", help="Path to input FITS image")
    input_group.add_argument("--fits-dir", help="Path to directory with FITS images")
    parser.add_argument("--out", default="results.csv", help="Path to output CSV")
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
        "--vsx-catalog",
        default="B/vsx/vsx",
        help="VizieR catalog name for VSX variable stars",
    )
    parser.add_argument(
        "--vsx-match-arcsec",
        type=float,
        default=2.0,
        help="Match radius in arcsec for APASS-to-VSX crossmatch",
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
        "--solver",
        choices=("auto", "remote", "local", "astap"),
        default="auto",
        help="WCS solver backend: auto (prefer local/astap), remote (astrometry.net API), local (solve-field), or astap (ASTAP)",
    )
    parser.add_argument(
        "--solve-field-cmd",
        default="solve-field",
        help="Command/path for local astrometry.net solve-field executable",
    )
    parser.add_argument(
        "--astap-cmd",
        default="astap",
        help="Command/path for the ASTAP solver executable (e.g. C:\\Program Files\\astap\\astap.exe)",
    )
    parser.add_argument(
        "--fit-seed-density",
        type=int,
        default=1,
        help="Seed density for multi-start logistic fit (>=1)",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Path to target CSV containing target names (one name per line or a target_name column)",
    )
    parser.add_argument(
        "--target-out",
        default=None,
        help="Path to target output CSV. Defaults to <out>_targets.csv",
    )
    parser.add_argument(
        "--target-match-arcsec",
        type=float,
        default=5.0,
        help="Match radius in arcsec for target-name coordinates to detected stars",
    )
    parser.add_argument(
        "--band",
        choices=("B", "V", "R"),
        default="V",
        help="Photometric band for calibration against APASS: B (Johnson), V (Johnson, default), or R (Sloan r')",
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


def _solve_wcs_remote(path: str, api_key: str, poll_interval: float, timeout: float) -> fits.Header:
    """Solve WCS via astrometry.net remote API with robust timeout handling."""
    ast = AstrometryNet()
    ast.api_key = api_key

    # Raise the HTTP-level read timeout so large uploads / slow responses
    # don't kill the connection before the server even starts solving.
    try:
        ast.TIMEOUT = 300          # individual HTTP request timeout (seconds)
    except Exception:
        pass

    start_time = time.time()
    max_wait = None if float(timeout) <= 0.0 else max(1.0, float(timeout))
    submission_id = None
    attempt = 0
    backoff = max(1.0, float(poll_interval))

    # ── helpers ────────────────────────────────────────────────────────
    def _remaining() -> float:
        if max_wait is None:
            return float("inf")
        return max(0.0, max_wait - (time.time() - start_time))

    def _timed_out() -> bool:
        return max_wait is not None and _remaining() <= 0

    def _extract_submission_id(exc: Exception):
        """Extract submission_id from an astroquery exception."""
        # astroquery's solve_from_image raises TimeoutError(msg, submission_id)
        if hasattr(exc, "args") and len(exc.args) > 1 and exc.args[1] is not None:
            try:
                return int(exc.args[1])
            except (TypeError, ValueError):
                pass
        # Fall back to regex search over the whole error text
        message = " ".join(str(part) for part in getattr(exc, "args", ()))
        for pattern in (
            r"submission[_ ]?id\s*[:=]?\s*(\d+)",
            r"subid\s*=\s*(\d+)",
        ):
            m = re.search(pattern, message, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except (TypeError, ValueError):
                    pass
        return None

    def _is_timeout_like(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(t in msg for t in (
            "timed out", "timeout", "solve timed out",
        ))

    def _is_transient(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(t in msg for t in (
            "connection aborted", "remotedisconnected",
            "remote end closed", "connection reset",
            "read timed out", "temporarily unavailable",
            "eof occurred", "broken pipe",
            "503", "502", "504",
        ))

    def _sleep(seconds: float) -> None:
        time.sleep(max(0.5, min(seconds, _remaining())))

    # ── main loop ─────────────────────────────────────────────────────
    while not _timed_out():
        attempt += 1
        remaining = _remaining()

        try:
            if submission_id is None:
                # First call: upload + solve.  Give it a generous share of
                # the remaining budget (up to 300 s) so that slow-but-
                # successful solves are not killed prematurely.
                call_timeout = max(30.0, min(300.0, remaining))
                print(
                    f"  [astrometry.net] Uploading {os.path.basename(path)} "
                    f"(attempt {attempt}, call timeout {call_timeout:.0f}s, "
                    f"remaining {remaining:.0f}s)..."
                )
                wcs_header = ast.solve_from_image(
                    path,
                    force_image_upload=True,
                    solve_timeout=call_timeout,
                )
            else:
                # We already own a submission – just keep monitoring.
                call_timeout = max(30.0, min(300.0, remaining))
                print(
                    f"  [astrometry.net] Monitoring submission {submission_id} "
                    f"(attempt {attempt}, call timeout {call_timeout:.0f}s, "
                    f"remaining {remaining:.0f}s)..."
                )
                monitor_fn = getattr(ast, "monitor_submission", None)
                if monitor_fn is None:
                    raise RuntimeError(
                        "astroquery version lacks monitor_submission(); "
                        "cannot continue polling existing submission"
                    )
                wcs_header = monitor_fn(
                    submission_id,
                    solve_timeout=call_timeout,
                )

            # ── validate result ───────────────────────────────────────
            if wcs_header is not None and len(wcs_header) > 0:
                print("  [astrometry.net] Solve succeeded.")
                return wcs_header

            # Empty / None header – treat as "not done yet"
            print("  [astrometry.net] Solve returned empty header, retrying...")
            _sleep(backoff)
            continue

        except TimeoutError as exc:
            sid = _extract_submission_id(exc)
            if sid is not None:
                submission_id = sid
                print(
                    f"  [astrometry.net] Timeout, but captured "
                    f"submission_id={submission_id}. Continuing to monitor..."
                )
            else:
                print(f"  [astrometry.net] Timeout (no submission_id): {exc}")
            _sleep(backoff)
            backoff = min(backoff * 1.5, 60.0)
            continue

        except Exception as exc:
            sid = _extract_submission_id(exc)
            if sid is not None:
                submission_id = sid

            if _is_timeout_like(exc):
                print(f"  [astrometry.net] Timeout-like error: {exc}")
                _sleep(backoff)
                backoff = min(backoff * 1.5, 60.0)
                continue

            if _is_transient(exc):
                print(
                    f"  [astrometry.net] Transient error (attempt {attempt}): {exc}"
                )
                _sleep(backoff)
                backoff = min(backoff * 1.5, 60.0)
                continue

            raise RuntimeError(f"Astrometry.net solve error: {exc}") from exc

    raise RuntimeError(
        f"Astrometry.net solve timed out after {timeout:.0f}s. "
        f"Last submission_id: "
        f"{submission_id if submission_id is not None else '(none)'}. "
        f"If the solve succeeds on the web GUI, try increasing --solve-timeout."
    )


def _solve_wcs_local(path: str, timeout: float, solve_field_cmd: str) -> fits.Header:
    solve_field_exe = shutil.which(solve_field_cmd) if os.path.basename(solve_field_cmd) == solve_field_cmd else solve_field_cmd
    if not solve_field_exe:
        raise RuntimeError(
            f"Local solver executable not found: {solve_field_cmd}. Install astrometry.net and ensure solve-field is in PATH"
        )

    with tempfile.TemporaryDirectory(prefix="astrometry_local_") as tmpdir:
        wcs_path = os.path.join(tmpdir, "solution.wcs")
        solved_path = os.path.join(tmpdir, "solution.solved")

        cmd = [
            solve_field_exe,
            path,
            "--overwrite",
            "--no-plots",
            "--dir",
            tmpdir,
            "--new-fits",
            "none",
            "--corr",
            "none",
            "--rdls",
            "none",
            "--match",
            "none",
            "--wcs",
            wcs_path,
            "--solved",
            solved_path,
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Local solve-field timed out after {timeout} s") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(f"Local solver command not found: {solve_field_cmd}") from exc

        if proc.returncode != 0 or not os.path.exists(wcs_path):
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = stderr if stderr else stdout
            raise RuntimeError(f"Local solve-field failed (exit {proc.returncode}). {details[:500]}")

        return fits.getheader(wcs_path)


def _solve_wcs_astap(path: str, timeout: float, astap_cmd: str) -> fits.Header:
    """Solve WCS using the local ASTAP solver."""
    astap_exe = (
        shutil.which(astap_cmd)
        if os.path.basename(astap_cmd) == astap_cmd
        else astap_cmd
    )
    if not astap_exe:
        astap_exe = astap_cmd  # let subprocess raise FileNotFoundError

    with tempfile.TemporaryDirectory(prefix="astap_") as tmpdir:
        tmp_fits = os.path.join(tmpdir, os.path.basename(path))
        shutil.copy2(path, tmp_fits)

        cmd = [astap_exe, "-f", tmp_fits, "-update"]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ASTAP solver timed out after {timeout} s") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"ASTAP executable not found: {astap_cmd}. "
                "Install ASTAP and use --astap-cmd to provide its path."
            ) from exc

        hdr = fits.getheader(tmp_fits)
        if "CRVAL1" in hdr and "CRVAL2" in hdr:
            return hdr

        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr if stderr else stdout
        raise RuntimeError(
            f"ASTAP failed to solve image (exit {proc.returncode}). {details[:500]}"
        )


def solve_wcs(
    path: str,
    api_key: str,
    poll_interval: float,
    timeout: float,
    solver: str = "auto",
    solve_field_cmd: str = "solve-field",
    astap_cmd: str = "astap",
) -> fits.Header:
    _ = poll_interval

    if solver == "local":
        return _solve_wcs_local(path, timeout, solve_field_cmd)

    if solver == "astap":
        return _solve_wcs_astap(path, timeout, astap_cmd)

    if solver == "remote":
        return _solve_wcs_remote(path, api_key, poll_interval, timeout)

    # auto: try astap, then solve-field, then remote
    for _try_local in (
        lambda: _solve_wcs_astap(path, timeout, astap_cmd),
        lambda: _solve_wcs_local(path, timeout, solve_field_cmd),
    ):
        try:
            return _try_local()
        except Exception:
            pass

    try:
        return _solve_wcs_remote(path, api_key, poll_interval, timeout)
    except Exception as exc:
        raise RuntimeError(
            f"Auto solver failed: tried astap, local solve-field, and remote. Last error: {exc}"
        ) from exc


def compute_search_radius(wcs: WCS, shape: tuple[int, int]) -> u.Quantity:
    ny, nx = shape
    corners = np.array(
        [[0, 0], [0, ny - 1], [nx - 1, 0], [nx - 1, ny - 1]], dtype=float
    )
    world = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
    center = wcs.pixel_to_world(nx / 2.0, ny / 2.0)
    separations = center.separation(world)
    return separations.max()


def _vizier_query_with_retry(
    vizier: Vizier,
    center: SkyCoord,
    radius: u.Quantity,
    catalog: str,
    max_retries: int = 5,
    initial_backoff: float = 5.0,
) -> Table:
    """Run a VizieR query_region with retry + exponential backoff."""
    backoff = initial_backoff
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = vizier.query_region(center, radius=radius, catalog=catalog)
            if not result:
                return Table()
            return result[0]
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = any(t in msg for t in (
                "connection", "timeout", "timed out", "refused",
                "reset", "503", "502", "504", "eof",
                "temporarily unavailable", "max retries",
                "remote end closed", "broken pipe",
            ))
            if not transient:
                raise
            print(
                f"  [VizieR] Transient error on attempt {attempt}/{max_retries}: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < max_retries:
                print(f"  [VizieR] Retrying in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 120.0)
    raise RuntimeError(
        f"VizieR query failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


def query_apass(center: SkyCoord, radius: u.Quantity, catalog: str) -> Table:
    vizier = Vizier(columns=["RAJ2000", "DEJ2000", "Vmag", "B-V", "Bmag", "r'mag", "e_Vmag", "ID"])
    vizier.ROW_LIMIT = -1
    return _vizier_query_with_retry(vizier, center, radius, catalog)


def query_vsx(center: SkyCoord, radius: u.Quantity, catalog: str) -> Table:
    vizier = Vizier(columns=["RAJ2000", "DEJ2000", "Name", "Type"])
    vizier.ROW_LIMIT = -1
    return _vizier_query_with_retry(vizier, center, radius, catalog)


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

    with warnings.catch_warnings():
        # photutils emits this when some PSF fits don't converge — the
        # "flags" column already captures that, so the console warning
        # is just noise.
        warnings.filterwarnings(
            "ignore",
            message=".*may not have converged.*",
            module=r"photutils\.psf.*",
        )
        return phot(data, init_params=init_params)


def resolve_flux_column(phot_table: Table) -> str:
    for name in ("flux_fit", "flux_0", "flux"):
        if name in phot_table.colnames:
            return name
    raise KeyError("No flux column found in photometry output")


def _forced_psf_flux(data: np.ndarray, x: float, y: float, fwhm: float) -> float:
    """PSF-fit the image at a fixed pixel position and return the fitted flux.

    This is used as a fallback for targets not detected by DAOStarFinder
    (e.g. a supernova embedded in a galaxy). Returns NaN if the fit fails
    or the position is too close to the image edge.
    """
    ny, nx = data.shape
    margin = fwhm * 3.0
    if x < margin or x >= nx - margin or y < margin or y >= ny - margin:
        return float("nan")

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
    init_params["x_0"] = [x]
    init_params["y_0"] = [y]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*may not have converged.*",
            module=r"photutils\.psf.*",
        )
        try:
            result = phot(data, init_params=init_params)
        except Exception:
            return float("nan")

    if len(result) == 0:
        return float("nan")

    for col in ("flux_fit", "flux_0", "flux"):
        if col in result.colnames:
            val = float(result[col][0])
            return val if val > 0.0 else float("nan")
    return float("nan")


def resolve_apass_column(apass: Table, candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in apass.colnames:
            return name
    raise KeyError(f"No {label} column found in APASS output. Columns: {apass.colnames}")


def extract_timestamp_jd_from_header(header: fits.Header) -> float:
    for key in ("DATE-OBS", "DATEOBS", "DATE_OBS", "DATE", "DATE-BEG", "DATE-END"):
        value = header.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            try:
                return float(Time(text, format="isot", scale="utc").jd)
            except Exception:
                try:
                    return float(Time(text, scale="utc").jd)
                except Exception:
                    continue
    return float("nan")


def _parse_inline_coord(coord_str: str) -> "SkyCoord | None":
    """Parse a sexagesimal 'RA DEC' string such as '15:15 57.21 +56° 18 32.2'.

    RA is interpreted as hour angle; Dec is signed degrees.
    Returns None if parsing fails.
    """
    coord_str = coord_str.strip()
    if not coord_str:
        return None
    tokens = re.split(r"\s+", coord_str)
    ra_tokens: list[str] = []
    dec_tokens: list[str] = []
    dec_started = False
    for token in tokens:
        if not dec_started and token and token[0] in "+-":
            dec_started = True
        if dec_started:
            dec_tokens.append(token)
        else:
            ra_tokens.append(token)
    if not ra_tokens or not dec_tokens:
        return None

    def _norm(parts: list[str]) -> str:
        sign = ""
        sub_parts: list[str] = []
        for i, p in enumerate(parts):
            # Strip any DMS decoration (°, ', ", and encoding variants)
            p = re.sub(r"[^0-9.:+\-]", "", p)
            if i == 0 and p and p[0] in "+-":
                sign = p[0]
                p = p[1:]
            sub_parts.extend(s for s in re.split(r":", p) if s)
        return sign + ":".join(sub_parts)

    ra_str = _norm(ra_tokens)
    dec_str = _norm(dec_tokens)
    try:
        return SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    except Exception:
        try:
            return SkyCoord(ra_str, dec_str, unit=(u.deg, u.deg))
        except Exception:
            return None


def _split_name_and_coord(raw: str) -> "tuple[str, SkyCoord | None]":
    """Split 'name = RA DEC' into (name, SkyCoord); plain name returns (name, None)."""
    if "=" in raw:
        left, _, right = raw.partition("=")
        return left.strip(), _parse_inline_coord(right.strip())
    return raw.strip(), None


def load_target_names(path: str) -> "tuple[list[str], dict[str, SkyCoord]]":
    """Return (names, pre_resolved) where pre_resolved maps name -> SkyCoord
    for entries given as 'name = RA DEC' in the file."""
    pre_resolved: dict[str, SkyCoord] = {}

    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if not sample.strip():
            return [], pre_resolved

        sniffer = csv.Sniffer()
        has_header = False
        try:
            has_header = sniffer.has_header(sample)
        except csv.Error:
            has_header = False

        if has_header:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return [], pre_resolved
            field_map = {name.lower().strip(): name for name in reader.fieldnames if name is not None}
            target_key = field_map.get("target_name") or field_map.get("name")
            if target_key is not None:
                names = []
                for row in reader:
                    value = row.get(target_key)
                    if value is None:
                        continue
                    raw = str(value).strip()
                    if not raw:
                        continue
                    name, coord = _split_name_and_coord(raw)
                    if name:
                        names.append(name)
                        if coord is not None:
                            pre_resolved[name] = coord
                return names, pre_resolved

        handle.seek(0)
        reader = csv.reader(handle)
        names = []
        for row in reader:
            if not row:
                continue
            # Rejoin all fields in case the coord string contains commas
            raw = ",".join(row).strip()
            if not raw or raw.lower() in {"target_name", "name"}:
                continue
            name, coord = _split_name_and_coord(raw)
            if name:
                names.append(name)
                if coord is not None:
                    pre_resolved[name] = coord
        return names, pre_resolved


def resolve_target_coordinates(
    target_names: list[str],
    pre_resolved: "dict[str, SkyCoord] | None" = None,
) -> "tuple[list[str], SkyCoord | None, list[str]]":
    if pre_resolved is None:
        pre_resolved = {}
    resolved_names: list[str] = []
    coords: list[SkyCoord] = []
    unresolved_names: list[str] = []

    for target_name in target_names:
        if target_name in pre_resolved:
            print(f"  [targets] Using inline coordinates for '{target_name}' (Simbad skipped).")
            resolved_names.append(target_name)
            coords.append(pre_resolved[target_name].icrs)
            continue
        try:
            coord = SkyCoord.from_name(target_name)
        except Exception:
            unresolved_names.append(target_name)
            continue
        resolved_names.append(target_name)
        coords.append(coord.icrs)

    if len(coords) == 0:
        return resolved_names, None, unresolved_names
    return resolved_names, SkyCoord(coords), unresolved_names


def build_target_output(
    output: Table,
    target_names: list[str],
    target_coords: SkyCoord,
    timestamp_jd: float,
    match_radius: u.Quantity,
    data: "np.ndarray | None" = None,
    wcs: "WCS | None" = None,
    fwhm: "float | None" = None,
    fit_method: "str | None" = None,
    fit_params: "dict | None" = None,
    band: str = "V",
) -> Table:
    cal_col = _BAND_CONFIG[band]["cal_col"]
    result = Table(
        names=("jd", "target_name", cal_col, "match_sep_arcsec"),
        dtype=(float, "U128", float, float),
    )
    if len(target_names) == 0:
        return result

    match_limit = match_radius.to_value(u.arcsec)

    # Rebuild calibration callable from stored params (used for forced photometry)
    calib_model = None
    if fit_method is not None and fit_params is not None:
        calib_model = _fit_model_from_params(fit_method, fit_params)

    star_coords = None
    v_mag_cal_values = None
    if len(output) > 0:
        star_coords = SkyCoord(np.asarray(output["RA"], dtype=float), np.asarray(output["dec"], dtype=float), unit=(u.deg, u.deg))
        v_mag_cal_values = _to_float_array(np.asarray(output[cal_col]))

    for target_name, target_coord in zip(target_names, target_coords):
        # --- Try PSF-catalog match first ---
        if star_coords is not None:
            idx, sep2d, _ = target_coord.match_to_catalog_sky(star_coords)
            sep_arcsec = sep2d.to_value(u.arcsec)
            sep_scalar = float(sep_arcsec.item()) if hasattr(sep_arcsec, "item") else float(sep_arcsec)
        else:
            sep_scalar = float("inf")

        if sep_scalar <= match_limit:
            vmag_cal = v_mag_cal_values[int(idx)]
            if not np.isfinite(vmag_cal):
                print(
                    f"  [targets] '{target_name}': matched star at {sep_scalar:.1f}\" "
                    f"has no calibrated magnitude ({cal_col}=NaN). Writing row anyway."
                )
            result.add_row((float(timestamp_jd), target_name, float(vmag_cal), sep_scalar))
            continue

        # --- Fallback: forced PSF photometry at the target position ---
        if data is not None and wcs is not None and fwhm is not None and calib_model is not None:
            # Check target is within the image footprint
            try:
                px, py = wcs.world_to_pixel(target_coord)
                px, py = float(px), float(py)
            except Exception:
                px, py = float("nan"), float("nan")

            ny, nx = data.shape
            if np.isfinite(px) and np.isfinite(py) and 0 <= px < nx and 0 <= py < ny:
                forced_flux = _forced_psf_flux(data, px, py, fwhm)
                if np.isfinite(forced_flux):
                    forced_mags = calib_model(np.array([forced_flux]))
                    vmag_forced = float(forced_mags[0]) if len(forced_mags) > 0 else float("nan")
                    if np.isfinite(vmag_forced):
                        print(
                            f"  [targets] '{target_name}': no PSF-catalog match "
                            f"(nearest {sep_scalar:.1f}\"), used forced PSF photometry \u2192 "
                            f"{cal_col}={vmag_forced:.3f}"
                        )
                        result.add_row((float(timestamp_jd), target_name, vmag_forced, -1.0))
                        continue
                    print(
                        f"  [targets] '{target_name}': forced PSF flux={forced_flux:.4f} "
                        f"is outside calibration range ({cal_col}=NaN)."
                    )
                else:
                    print(
                        f"  [targets] '{target_name}': forced PSF fit at pixel "
                        f"({px:.1f}, {py:.1f}) failed (flux=NaN). "
                        f"Source may be too faint or overwhelmed by galaxy background."
                    )
            else:
                print(
                    f"  [targets] '{target_name}': coordinates map to pixel "
                    f"({px:.1f}, {py:.1f}) which is outside the image."
                )
        else:
            print(
                f"  [targets] '{target_name}': nearest star is {sep_scalar:.1f}\" away "
                f"(limit {match_limit:.1f}\"). No match — target may be outside the field "
                f"or --target-match-arcsec is too small."
            )

    return result


def create_empty_target_table(band: str = "V") -> Table:
    cal_col = _BAND_CONFIG[band]["cal_col"]
    return Table(
        names=("jd", "target_name", cal_col, "match_sep_arcsec"),
        dtype=(float, "U128", float, float),
    )


def list_fits_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        raise ValueError(f"Not a directory: {directory}")

    fits_paths: list[str] = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        lower = entry.lower()
        if lower.endswith((".fits", ".fit", ".fts")):
            fits_paths.append(full_path)

    fits_paths.sort()
    return fits_paths


def process_single_fits(
    fits_path: str,
    args: argparse.Namespace,
    api_key: str,
    cached_apass: Table | None = None,
    cached_vsx: Table | None = None,
) -> tuple[Table, str, dict[str, float | str], float, Table, Table]:
    """Process one FITS file.

    Returns (output, fit_method, fit_params, timestamp_jd, apass, vsx).
    Callers can pass cached_apass / cached_vsx to avoid redundant VizieR
    queries when processing multiple files of the same field.
    """
    data, header = load_image(fits_path)
    timestamp_jd = extract_timestamp_jd_from_header(header)

    wcs_header = solve_wcs(
        fits_path,
        api_key,
        args.poll_interval,
        args.solve_timeout,
        solver=args.solver,
        solve_field_cmd=args.solve_field_cmd,
        astap_cmd=args.astap_cmd,
    )
    header.update(wcs_header)
    # Ensure NAXIS reflects the image dimensions so astropy doesn't warn
    # about "WCS has more axes than the image".
    if data.ndim == 2 and header.get("NAXIS", 0) < 2:
        header["NAXIS"] = 2
        header["NAXIS1"] = data.shape[1]
        header["NAXIS2"] = data.shape[0]
    wcs = WCS(header)

    radius = compute_search_radius(wcs, data.shape)
    center = wcs.pixel_to_world(data.shape[1] / 2.0, data.shape[0] / 2.0)

    if cached_apass is not None:
        apass = cached_apass
    else:
        apass = query_apass(center, radius, args.apass_catalog)

    if cached_vsx is not None:
        vsx = cached_vsx
    else:
        vsx = query_vsx(center, radius, args.vsx_catalog)

    sources = detect_sources(data, args.fwhm, args.threshold_sigma, args.max_stars)
    phot_table = run_psf_photometry(data, sources, args.fwhm)

    output = build_output(
        phot_table,
        wcs,
        apass,
        vsx,
        args.match_arcsec * u.arcsec,
        args.vsx_match_arcsec * u.arcsec,
        band=args.band,
    )
    fit_method, fit_params = add_calibrated_vmag(output, seed_density=args.fit_seed_density, band=args.band)
    return output, fit_method, fit_params, timestamp_jd, apass, vsx, data, wcs


def _logistic_flux_eval(mag: np.ndarray, flux_low: float, flux_high: float, m0: float, slope: float) -> np.ndarray:
    z = np.clip(slope * (mag - m0), -700.0, 700.0)
    return flux_low + (flux_high - flux_low) / (1.0 + np.exp(z))


# Signature required by scipy.optimize.curve_fit (positional scalar args, no type hints)
def _logistic_flux_eval_4(mag, flux_low, flux_high, m0, slope):
    z = np.clip(slope * (mag - m0), -700.0, 700.0)
    return flux_low + (flux_high - flux_low) / (1.0 + np.exp(z))


def _fit_model_from_params(fit_method: str, fit_params: dict):
    """Reconstruct the calibration callable (flux -> v_mag) from stored fit_params."""
    if fit_method == "inverse_logistic_from_flux_numeric":
        flux_low = float(fit_params.get("fit_flux_low", np.nan))
        flux_high = float(fit_params.get("fit_flux_high", np.nan))
        m0 = float(fit_params.get("fit_m0", np.nan))
        slope = float(fit_params.get("fit_slope", np.nan))
        if not all(np.isfinite([flux_low, flux_high, m0, slope])):
            return None

        def _logistic_model(flux_values):
            return _invert_flux_to_vmag_numeric(
                np.asarray(flux_values, dtype=float),
                flux_low, flux_high, m0, slope,
                mag_lo=m0 - 12.0, mag_hi=m0 + 12.0,
            )
        return _logistic_model

    if fit_method == "linear_log_flux_fallback":
        slope = float(fit_params.get("fit_slope", np.nan))
        intercept = float(fit_params.get("fit_linear_intercept", np.nan))
        if not all(np.isfinite([slope, intercept])):
            return None

        def _linear_model(flux_values):
            return slope * np.log10(np.maximum(np.asarray(flux_values, dtype=float), 1e-30)) + intercept
        return _linear_model

    return None


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
        # Seed flux_high well above the data so the logistic asymptote
        # is not forced into the observed range (which produces a degenerate fit).
        flux_high_seed = max(flux_high_seed, y_max_obs * 5.0)

        seed_density = max(1, int(seed_density))
        mag_span = max(0.1, float(np.nanpercentile(x_mag, 90) - np.nanpercentile(x_mag, 10)))
        q_count = 3 + 2 * seed_density
        quantiles = np.linspace(20.0, 80.0, q_count)
        m0_seeds = np.nanpercentile(x_mag, quantiles)
        slope_base = max(0.08, 4.0 / mag_span)
        slope_factors = np.linspace(0.5, 1.8, 2 + seed_density)
        slope_seeds = [min(3.5, max(0.05, float(slope_base * factor))) for factor in slope_factors]

        # Bounds: flux_high must sit above the data (prevents degenerate asymptote).
        # scipy.optimize.curve_fit with method='trf' enforces bounds strictly.
        lb = [0.0,               y_max_obs * 1.5, mag_min - 2.0, 0.03]
        ub = [y_min_obs * 1.02,  y_max_obs * 500.0, mag_max + 2.0, 4.0]
        # Clamp seed into bounds
        p0_flux_low  = float(np.clip(flux_low_seed,  lb[0], ub[0]))
        p0_flux_high = float(np.clip(flux_high_seed, lb[1], ub[1]))

        best_popt = None
        best_score = np.inf

        for m0_seed in m0_seeds:
            for slope_seed in slope_seeds:
                p0 = [p0_flux_low, p0_flux_high, float(m0_seed), float(slope_seed)]
                try:
                    popt, _ = _scipy_curve_fit(
                        _logistic_flux_eval_4,
                        x_mag, y_flux,
                        p0=p0,
                        bounds=(lb, ub),
                        method='trf',
                        max_nfev=5000,
                    )
                except Exception:
                    continue
                pred = _logistic_flux_eval_4(x_mag, *popt)
                if not np.all(np.isfinite(pred)):
                    continue
                score = float(np.mean((pred - y_flux) ** 2))
                if score < best_score:
                    best_score = score
                    best_popt = popt

        if best_popt is not None:
            flux_low, flux_high, m0, slope = (float(v) for v in best_popt)

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


def plot_vmag_vs_flux(output: Table, plot_path: str, seed_density: int = 1, band: str = "V") -> None:
    import matplotlib.pyplot as plt

    mag_col = _BAND_CONFIG[band]["out_col"]
    band_label = _BAND_CONFIG[band]["label"]

    flux_values = []
    vmag_values = []
    bv_values = []

    for flux, vmag, bv in zip(output["flux"], output[mag_col], output["b-v"]):
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
    for flux, vmag in zip(output["flux"], output[mag_col]):
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
        ax.set_ylabel(mag_col)
        ax.set_title(f"APASS {band_label} vs PSF flux (color: b-v)")
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
        ax.text(0.5, 0.5, f"No matched APASS {band_label} magnitudes with b-v", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_vmagcal_vs_vmag(output: Table, plot_path: str, band: str = "V") -> None:
    import matplotlib.pyplot as plt

    mag_col = _BAND_CONFIG[band]["out_col"]
    cal_col = _BAND_CONFIG[band]["cal_col"]

    vmag_ref = []
    vmag_cal = []
    bv_values = []

    for vmag, vmagc, bv in zip(output[mag_col], output[cal_col], output["b-v"]):
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
        ax.plot([line_min, line_max], [line_min, line_max], "k--", lw=1.2, label=f"{mag_col} = {cal_col}")

        ax.set_xlim(line_min, line_max)
        ax.set_ylim(line_min, line_max)
        ax.set_xlabel(f"{mag_col} (APASS)")
        ax.set_ylabel(f"{cal_col} (calibrated)")
        ax.set_title(f"Calibration QC: {cal_col} vs {mag_col}")
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
        ax.text(0.5, 0.5, f"No stars with valid {mag_col}, {cal_col} and b-v", ha="center", va="center")
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


def add_calibrated_vmag(output: Table, seed_density: int = 1, band: str = "V") -> tuple[str, dict[str, float | str]]:
    mag_col = _BAND_CONFIG[band]["out_col"]
    cal_col = _BAND_CONFIG[band]["cal_col"]

    if len(output) == 0:
        output[cal_col] = np.array([], dtype=float)
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
    vmag_arr = _to_float_array(np.asarray(output[mag_col]))

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

    output[cal_col] = vmag_cal
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
    vsx: Table,
    match_radius: u.Quantity,
    vsx_match_radius: u.Quantity,
    band: str = "V",
) -> Table:
    mag_col = _BAND_CONFIG[band]["out_col"]
    cal_col = _BAND_CONFIG[band]["cal_col"]

    if len(phot_table) == 0:
        return Table(
            names=(
                "RA",
                "dec",
                "flux",
                mag_col,
                "b-v",
                "is_vsx_variable",
                "vsx_name",
                cal_col,
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
    out[mag_col] = np.full(len(phot_table), "", dtype=object)
    out["b-v"] = np.full(len(phot_table), "", dtype=object)
    out["is_vsx_variable"] = np.full(len(phot_table), False, dtype=bool)
    out["vsx_name"] = np.full(len(phot_table), "", dtype=object)

    if len(apass) > 0:
        ra_col = resolve_apass_column(apass, ("RAJ2000", "RA_ICRS", "RAdeg", "RA"), "RA")
        dec_col = resolve_apass_column(apass, ("DEJ2000", "DE_ICRS", "DEdeg", "DEC", "Dec"), "DEC")
        vmag_col = resolve_apass_column(apass, _BAND_CONFIG[band]["apass_mag_candidates"], f"{band} magnitude")

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

        variable_apass = np.zeros(len(apass), dtype=bool)
        apass_vsx_name = np.full(len(apass), "", dtype=object)
        if len(vsx) > 0:
            try:
                vsx_ra_col = resolve_apass_column(vsx, ("RAJ2000", "RA_ICRS", "RAdeg", "RA"), "VSX RA")
                vsx_dec_col = resolve_apass_column(vsx, ("DEJ2000", "DE_ICRS", "DEdeg", "DEC", "Dec"), "VSX DEC")
                vsx_coords = SkyCoord(vsx[vsx_ra_col], vsx[vsx_dec_col], unit=(u.deg, u.deg))
                vsx_idx, vsx_sep2d, _ = apass_coords.match_to_catalog_sky(vsx_coords)
                variable_apass = vsx_sep2d <= vsx_match_radius
                if "Name" in vsx.colnames:
                    apass_vsx_name[variable_apass] = np.asarray(vsx["Name"][vsx_idx][variable_apass], dtype=str)
                elif "VSX" in vsx.colnames:
                    apass_vsx_name[variable_apass] = np.asarray(vsx["VSX"][vsx_idx][variable_apass], dtype=str)
            except Exception:
                variable_apass = np.zeros(len(apass), dtype=bool)
                apass_vsx_name = np.full(len(apass), "", dtype=object)

        idx, sep2d, _ = sky.match_to_catalog_sky(apass_coords)
        matched = sep2d <= match_radius
        not_variable = ~variable_apass[idx]
        matched_non_variable = matched & not_variable
        matched_variable = matched & (~not_variable)

        matched_indices = np.where(matched_non_variable)[0]
        variable_indices = np.where(matched_variable)[0]
        out[mag_col][matched_indices] = np.asarray(apass[vmag_col][idx][matched_non_variable], dtype=str)
        out["is_vsx_variable"][variable_indices] = True
        out["vsx_name"][variable_indices] = np.asarray(apass_vsx_name[idx][matched_variable], dtype=str)
        if bv_col is not None:
            out["b-v"][matched_indices] = np.asarray(apass[bv_col][idx][matched_non_variable], dtype=str)
        elif use_computed_bv and bmag_col is not None:
            bmag_values = np.asarray(apass[bmag_col][idx][matched_non_variable], dtype=float)
            vmag_values = np.asarray(apass[vmag_col][idx][matched_non_variable], dtype=float)
            out["b-v"][matched_indices] = np.asarray(bmag_values - vmag_values, dtype=str)

    return out


def main() -> int:
    args = parse_args()
    args.fit_seed_density = max(1, int(args.fit_seed_density))

    api_key = os.environ.get(args.api_key_env)
    if not api_key and args.solver not in ("local", "astap"):
        print(f"Missing astrometry.net API key in {args.api_key_env}")
        return 2
    api_key = api_key or ""

    if args.fits_dir:
        if not args.targets:
            print("In --fits-dir mode, --targets is required")
            return 2

        fits_paths = list_fits_files(args.fits_dir)
        if len(fits_paths) == 0:
            print(f"No FITS files found in {args.fits_dir}")
            return 2

        target_names, pre_resolved = load_target_names(args.targets)
        resolved_names, target_coords, unresolved_names = resolve_target_coordinates(target_names, pre_resolved)
        target_out_path = args.target_out if args.target_out else f"{os.path.splitext(args.out)[0]}_targets.csv"

        combined_target_rows = create_empty_target_table(band=args.band)
        cached_apass: Table | None = None
        cached_vsx: Table | None = None

        for fits_path in fits_paths:
            output, _fm, _fp, timestamp_jd, cached_apass, cached_vsx, _data, _wcs = process_single_fits(
                fits_path, args, api_key,
                cached_apass=cached_apass,
                cached_vsx=cached_vsx,
            )
            if target_coords is None:
                continue
            file_target_rows = build_target_output(
                output=output,
                target_names=resolved_names,
                target_coords=target_coords,
                timestamp_jd=timestamp_jd,
                match_radius=args.target_match_arcsec * u.arcsec,
                data=_data,
                wcs=_wcs,
                fwhm=args.fwhm,
                fit_method=_fm,
                fit_params=_fp,
                band=args.band,
            )
            if len(file_target_rows) > 0:
                if len(combined_target_rows) == 0:
                    combined_target_rows = file_target_rows
                else:
                    combined_target_rows = vstack([combined_target_rows, file_target_rows])

        combined_target_rows.write(target_out_path, format="csv", overwrite=True)
        print(f"Processed {len(fits_paths)} FITS files from {args.fits_dir}")
        print(f"Wrote {len(combined_target_rows)} target rows to {target_out_path}")
        if len(unresolved_names) > 0:
            print(f"Warning: {len(unresolved_names)} target names could not be resolved")
        return 0

    output, fit_method, fit_params, timestamp_jd, _, _, _data, _wcs = process_single_fits(args.fits, args, api_key)
    output.write(args.out, format="csv", overwrite=True)

    target_rows = None
    target_out_path = None
    unresolved_names: list[str] = []
    if args.targets:
        target_names, pre_resolved = load_target_names(args.targets)
        resolved_names, target_coords, unresolved_names = resolve_target_coordinates(target_names, pre_resolved)
        target_out_path = args.target_out if args.target_out else f"{os.path.splitext(args.out)[0]}_targets.csv"
        if target_coords is None:
            target_rows = create_empty_target_table(band=args.band)
        else:
            target_rows = build_target_output(
                output=output,
                target_names=resolved_names,
                target_coords=target_coords,
                timestamp_jd=timestamp_jd,
                match_radius=args.target_match_arcsec * u.arcsec,
                data=_data,
                wcs=_wcs,
                fwhm=args.fwhm,
                fit_method=fit_method,
                fit_params=fit_params,
                band=args.band,
            )
        target_rows.write(target_out_path, format="csv", overwrite=True)

    plot_path = args.plot if args.plot else f"{os.path.splitext(args.out)[0]}_vmag_vs_flux.png"
    plot_vmag_vs_flux(output, plot_path, seed_density=args.fit_seed_density, band=args.band)
    qc_plot_path = f"{os.path.splitext(args.out)[0]}_vmagcal_vs_vmag.png"
    plot_vmagcal_vs_vmag(output, qc_plot_path, band=args.band)

    print(f"Wrote {len(output)} rows to {args.out}")
    print(f"{_BAND_CONFIG[args.band]['label']} calibration model: {fit_method}")
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
    if target_rows is not None and target_out_path is not None:
        print(f"Wrote {len(target_rows)} target rows to {target_out_path}")
        if len(unresolved_names) > 0:
            print(f"Warning: {len(unresolved_names)} target names could not be resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())

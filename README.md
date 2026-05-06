# psf_photometry
An easy to use python script for PSF photometry.

## Usage

Set your astrometry.net API key in an environment variable and run the script:

```powershell

python psf_photometry.py --fits path\to\image.fits --out results.csv
```

Optional: increase nonlinear fit robustness with more multi-start seeds:

```powershell
python psf_photometry.py --fits path\to\image.fits --out results.csv --fit-seed-density 2
```

Optional: pass a target name list and write calibrated target magnitudes:

```powershell
python psf_photometry.py --fits path\to\image.fits --out results.csv --targets target.csv
```

Optional: process all FITS files in a directory and write one combined target result file:

```powershell
python psf_photometry.py --fits-dir path\to\fits_folder --targets target.csv --out results.csv
```

In `--fits-dir` mode, all FITS files in the directory are processed automatically.
Only the combined target output (`results_targets.csv` by default) is written.
Per-image detailed outputs (`results.csv`) and plot images are not written in this mode.

Optional: use a local astrometry.net solver (`solve-field`) instead of the online API:

```powershell
python psf_photometry.py --fits path\to\image.fits --out results.csv --solver local
```

Optional: use the local [ASTAP](https://www.hnsky.org/astap.htm) solver:

```powershell
python psf_photometry.py --fits path\to\image.fits --out results.csv --solver astap
```

If `astap` is not on your system PATH, supply the full path with `--astap-cmd`:

```powershell
python psf_photometry.py --fits path\to\image.fits --out results.csv --solver astap --astap-cmd "C:\Program Files\astap\astap.exe"
```

`--solver auto` (default) tries ASTAP first, then local `solve-field`, then falls back to the remote API.
You can set a custom executable path with `--solve-field-cmd` (astrometry.net) or `--astap-cmd` (ASTAP).

If the remote astrometry API occasionally times out although the web UI later shows a solved job,
the script now continues polling the same submission automatically. For slow solves, increase
`--solve-timeout` (for example `--solve-timeout 1800`) and optionally use `--solver remote`.
Set `--solve-timeout 0` to disable the overall timeout and keep polling until the submission resolves.
Transient network errors like `Connection aborted` / `RemoteDisconnected` are retried automatically.

The generated `results.csv` contains one row per PSF-photometry source with these columns:

1. `RA` (right ascension in degrees from solved WCS)
2. `dec` (declination in degrees from solved WCS)
3. `flux` (PSF photometry flux)
4. `v_mag` (APASS V magnitude if matched, otherwise empty)
5. `b-v` (APASS B-V color index if matched, otherwise empty)
6. `is_vsx_variable` (`True` if matched APASS counterpart is identified as variable in VSX)
7. `vsx_name` (name of matched VSX entry for variable stars, otherwise empty)
8. `v_mag_cal` (calibrated V magnitude from nonlinear regression of `flux` vs `v_mag`)
9. `fit_model` (used calibration model)
10. `fit_flux_low`, `fit_flux_high`, `fit_m0`, `fit_slope` (4-parameter sigmoid fit values)
11. `fit_linear_intercept` (only set for linear fallback)
12. `fit_seed_density`, `fit_n_refstars` (fit configuration and number of APASS reference stars)

For differential photometry, the script fits a 4-parameter sigmoid (logistic) in the
forward direction `flux(v_mag)` and then computes `v_mag(flux)` by numerical
inversion (bisection). This provides the required inverse mapping for calibration of
all detected stars (including those without APASS data). If the nonlinear fit cannot
be computed, a linear fallback in `log10(flux)` is used.

Additionally, a PNG plot `v_mag` vs `flux` is written to `<out>_vmag_vs_flux.png` by default.
Plot points are colored by `b-v` using a red-to-blue palette, with larger `b-v` values shown in blue.
You can set a custom plot path with `--plot path\to\plot.png`.

All APASS stars in the solved field are crossmatched against the VSX catalog.
Stars identified as variables in VSX are excluded from calibration and from the plots.
You can configure this with `--vsx-catalog` and `--vsx-match-arcsec`.

For quality control, an additional plot `<out>_vmagcal_vs_vmag.png` is written.
It shows `v_mag_cal` vs APASS `v_mag` for stars with known APASS `v_mag`, includes the
reference line `v_mag = v_mag_cal` (Winkelhalbierende), and colors points by `b-v`.

If `--targets target.csv` is provided, the script resolves target names to sky coordinates,
crossmatches them with detected stars in `results.csv`, and writes `<out>_targets.csv`
with rows in this format:

1. `jd` (Julian Date derived from FITS header time, e.g. `DATE-OBS`)
2. `target_name`
3. `v_mag_cal`
4. `match_sep_arcsec` (separation between resolved target position and matched detected star)

`target.csv` can be either one target name per line or a CSV with a `target_name`/`name` column.

## Dependencies

- astropy
- astroquery
- photutils
- numpy
- matplotlib

Install with:

```powershell
python -m pip install astropy astroquery photutils numpy matplotlib
```

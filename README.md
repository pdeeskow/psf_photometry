# psf_photometry
An easy to use python script for PSF photometry.

## Usage

Set your astrometry.net API key in an environment variable and run the script:

```powershell
$env:ASTROMETRY_API_KEY = "your_key_here"
python psf_photometry.py --fits path\to\image.fits --out results.csv
```

Optional: increase nonlinear fit robustness with more multi-start seeds:

```powershell
python psf_photometry.py --fits path\to\image.fits --out results.csv --fit-seed-density 2
```

The generated `results.csv` contains one row per PSF-photometry source with these columns:

1. `RA` (right ascension in degrees from solved WCS)
2. `dec` (declination in degrees from solved WCS)
3. `flux` (PSF photometry flux)
4. `v_mag` (APASS V magnitude if matched, otherwise empty)
5. `b-v` (APASS B-V color index if matched, otherwise empty)
6. `v_mag_cal` (calibrated V magnitude from nonlinear regression of `flux` vs `v_mag`)
7. `fit_model` (used calibration model)
8. `fit_flux_low`, `fit_flux_high`, `fit_m0`, `fit_slope` (4-parameter sigmoid fit values)
9. `fit_linear_intercept` (only set for linear fallback)
10. `fit_seed_density`, `fit_n_refstars` (fit configuration and number of APASS reference stars)

For differential photometry, the script fits a 4-parameter sigmoid (logistic) in the
forward direction `flux(v_mag)` and then computes `v_mag(flux)` by numerical
inversion (bisection). This provides the required inverse mapping for calibration of
all detected stars (including those without APASS data). If the nonlinear fit cannot
be computed, a linear fallback in `log10(flux)` is used.

Additionally, a PNG plot `v_mag` vs `flux` is written to `<out>_vmag_vs_flux.png` by default.
Plot points are colored by `b-v` using a red-to-blue palette, with larger `b-v` values shown in blue.
You can set a custom plot path with `--plot path\to\plot.png`.

For quality control, an additional plot `<out>_vmagcal_vs_vmag.png` is written.
It shows `v_mag_cal` vs APASS `v_mag` for stars with known APASS `v_mag`, includes the
reference line `v_mag = v_mag_cal` (Winkelhalbierende), and colors points by `b-v`.

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

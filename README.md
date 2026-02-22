# psf_photometry
An easy to use python script for PSF photometry.

## Usage

Set your astrometry.net API key in an environment variable and run the script:

```powershell
$env:ASTROMETRY_API_KEY = "your_key_here"
python psf_photometry.py --fits path\to\image.fits --out results.csv
```

The generated `results.csv` contains one row per PSF-photometry source with these columns:

1. `RA` (right ascension in degrees from solved WCS)
2. `dec` (declination in degrees from solved WCS)
3. `flux` (PSF photometry flux)
4. `v_mag` (APASS V magnitude if matched, otherwise empty)
5. `b-v` (APASS B-V color index if matched, otherwise empty)

Additionally, a PNG plot `v_mag` vs `flux` is written to `<out>_vmag_vs_flux.png` by default.
Plot points are colored by `b-v` using a red-to-blue palette, with larger `b-v` values shown in blue.
You can set a custom plot path with `--plot path\to\plot.png`.

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

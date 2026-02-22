# psf_photometry
An easy to use python script for PSF photometry.

## Usage

Set your astrometry.net API key in an environment variable and run the script:

```powershell
$env:ASTROMETRY_API_KEY = "your_key_here"
python psf_photometry.py --fits path\to\image.fits --out results.csv
```

## Dependencies

- astropy
- astroquery
- photutils
- numpy

Install with:

```powershell
python -m pip install astropy astroquery photutils numpy
```

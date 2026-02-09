# PSF Photometry

An easy to use Python script for PSF photometry on FITS files with astrometry and catalog matching.

## Features

- Reads FITS files
- Performs plate solving with astrometry.net
- Matches identified stars with the APASS catalog
- Performs PSF photometry on detected sources
- Outputs results to CSV with source ID, flux, and V band magnitude from APASS

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python psf_photometry.py input_file.fits
```

This will create a `photometry_results.csv` file with the results.

### Command-line Options

- `fits_file`: Input FITS file (required)
- `-o`, `--output`: Output CSV file path (default: `photometry_results.csv`)
- `--fwhm`: Full-width at half-maximum of stars in pixels (default: 3.0)
- `--threshold`: Detection threshold in sigma (default: 5.0)
- `--tolerance`: Catalog matching tolerance in arcseconds (default: 2.0)
- `--api-key`: Astrometry.net API key (optional, for plate solving)

### Examples

Specify custom output file and FWHM:

```bash
python psf_photometry.py myimage.fits -o results.csv --fwhm 4.5
```

Use astrometry.net with API key:

```bash
python psf_photometry.py myimage.fits --api-key YOUR_API_KEY
```

Adjust detection parameters:

```bash
python psf_photometry.py myimage.fits --threshold 3.0 --tolerance 1.5
```

## Output Format

The output CSV file contains three columns:

- `id`: Source identifier (index of detected source)
- `flux`: PSF-fitted flux of the source
- `Vmag`: V band magnitude from APASS catalog

## Requirements

See `requirements.txt` for the full list of dependencies. Key packages include:

- astropy: FITS file handling and WCS
- photutils: PSF photometry
- astroquery: Astrometry.net and APASS catalog access
- pandas: CSV output
- numpy, scipy: Numerical computations

## Notes

- If plate solving with astrometry.net fails, the script will attempt to use WCS information from the FITS header
- The script requires an internet connection to query the APASS catalog and optionally for astrometry.net plate solving
- An astrometry.net API key is optional but recommended for better plate solving results

## License

Apache License 2.0

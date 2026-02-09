#!/usr/bin/env python3
"""
PSF Photometry Script

This script performs PSF photometry on a FITS file with the following steps:
1. Read FITS file
2. Perform plate solving with astrometry.net
3. Match identified stars with APASS catalog
4. Perform PSF photometry
5. Write results to CSV with id, flux, and V band magnitude from APASS
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils.psf import CircularGaussianSigmaPRF, SourceGrouper
from photutils.background import MMMBackground, MADStdBackgroundRMS, LocalBackground
from photutils.psf import PSFPhotometry
from astroquery.astrometry_net import AstrometryNet
from astroquery.vizier import Vizier


def read_fits(fits_path):
    """
    Read a FITS file and return the data and header.
    
    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file
        
    Returns
    -------
    data : numpy.ndarray
        Image data
    header : astropy.io.fits.Header
        FITS header
    """
    print(f"Reading FITS file: {fits_path}")
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header


def solve_plate(fits_path, api_key=None):
    """
    Perform plate solving using astrometry.net.
    
    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file
    api_key : str, optional
        Astrometry.net API key
        
    Returns
    -------
    wcs : astropy.wcs.WCS
        World Coordinate System solution
    """
    print("Performing plate solving with astrometry.net...")
    
    ast = AstrometryNet()
    if api_key:
        ast.api_key = api_key
    
    try:
        # Try to solve the field
        wcs_header = ast.solve_from_image(fits_path, force_image_upload=True)
        wcs = WCS(wcs_header)
        print("Plate solving successful!")
        return wcs
    except Exception as e:
        print(f"Plate solving failed: {e}")
        print("Attempting to use existing WCS from FITS header...")
        data, header = read_fits(fits_path)
        try:
            wcs = WCS(header)
            if wcs.has_celestial:
                print("Using WCS from FITS header")
                return wcs
        except Exception:
            pass
        raise ValueError("Could not obtain WCS solution")


def detect_sources(data, fwhm=3.0, threshold=5.0):
    """
    Detect sources in the image using DAOStarFinder.
    
    Parameters
    ----------
    data : numpy.ndarray
        Image data
    fwhm : float
        Full-width at half-maximum of stars in pixels
    threshold : float
        Detection threshold in sigma
        
    Returns
    -------
    sources : astropy.table.Table
        Detected sources
    """
    print(f"Detecting sources (FWHM={fwhm}, threshold={threshold} sigma)...")
    
    # Calculate background statistics
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    # Detect sources
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data - median)
    
    if sources is None:
        raise ValueError("No sources detected")
    
    print(f"Detected {len(sources)} sources")
    return sources


def query_apass(wcs, image_shape, radius=None):
    """
    Query APASS catalog for stars in the field of view.
    
    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World Coordinate System
    image_shape : tuple
        Shape of the image (ny, nx)
    radius : astropy.units.Quantity, optional
        Search radius. If None, estimated from image size
        
    Returns
    -------
    apass_catalog : astropy.table.Table
        APASS catalog entries
    """
    print("Querying APASS catalog...")
    
    # Get center coordinates
    ny, nx = image_shape
    center_x, center_y = nx / 2, ny / 2
    center_coord = wcs.pixel_to_world(center_x, center_y)
    
    # Estimate search radius if not provided
    if radius is None:
        corner_coord = wcs.pixel_to_world(0, 0)
        radius = center_coord.separation(corner_coord) * 1.5
    
    print(f"Searching around RA={center_coord.ra.deg:.4f}, Dec={center_coord.dec.deg:.4f}")
    print(f"Search radius: {radius.to(u.arcmin):.2f}")
    
    # Query Vizier for APASS catalog (II/336/apass9)
    v = Vizier(columns=['*', 'RAJ2000', 'DEJ2000', 'Vmag', 'e_Vmag'], 
               row_limit=-1)
    
    try:
        result = v.query_region(center_coord, radius=radius, catalog='II/336/apass9')
        if len(result) == 0:
            raise ValueError("No APASS catalog entries found")
        
        apass_catalog = result[0]
        print(f"Found {len(apass_catalog)} APASS catalog entries")
        return apass_catalog
    except Exception as e:
        print(f"APASS query failed: {e}")
        raise


def match_catalogs(sources, wcs, apass_catalog, tolerance=2.0):
    """
    Match detected sources with APASS catalog.
    
    Parameters
    ----------
    sources : astropy.table.Table
        Detected sources with x, y positions
    wcs : astropy.wcs.WCS
        World Coordinate System
    apass_catalog : astropy.table.Table
        APASS catalog
    tolerance : float
        Matching tolerance in arcseconds
        
    Returns
    -------
    matched : astropy.table.Table
        Matched sources with APASS data
    """
    print(f"Matching sources with APASS catalog (tolerance={tolerance} arcsec)...")
    
    # Convert pixel coordinates to sky coordinates
    source_coords = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
    
    # Convert APASS coordinates
    apass_coords = SkyCoord(ra=apass_catalog['RAJ2000'], 
                           dec=apass_catalog['DEJ2000'], 
                           unit=(u.deg, u.deg))
    
    # Match catalogs
    idx, d2d, d3d = source_coords.match_to_catalog_sky(apass_coords)
    
    # Filter by tolerance
    tolerance_angle = tolerance * u.arcsec
    matched_mask = d2d < tolerance_angle
    
    print(f"Matched {matched_mask.sum()} sources with APASS catalog")
    
    if matched_mask.sum() == 0:
        raise ValueError("No matches found between sources and APASS catalog")
    
    # Create matched table
    matched = Table()
    matched['source_id'] = np.arange(len(sources))[matched_mask]
    matched['x'] = sources['xcentroid'][matched_mask]
    matched['y'] = sources['ycentroid'][matched_mask]
    matched['flux'] = sources['flux'][matched_mask]
    matched['apass_id'] = idx[matched_mask]
    matched['separation'] = d2d[matched_mask].to(u.arcsec)
    matched['Vmag'] = apass_catalog['Vmag'][idx[matched_mask]]
    
    return matched


def perform_psf_photometry(data, sources, fwhm=3.0):
    """
    Perform PSF photometry on detected sources.
    
    Parameters
    ----------
    data : numpy.ndarray
        Image data
    sources : astropy.table.Table
        Detected sources
    fwhm : float
        Full-width at half-maximum in pixels
        
    Returns
    -------
    photometry : astropy.table.Table
        PSF photometry results
    """
    print("Performing PSF photometry...")
    
    # Set up PSF model
    sigma = fwhm / 2.355  # Convert FWHM to sigma
    psf_model = CircularGaussianSigmaPRF(sigma=sigma)
    
    # Set up local background estimator
    # Use annulus around each source for background estimation
    bkg_estimator = LocalBackground(
        inner_radius=fwhm * 2,
        outer_radius=fwhm * 3,
        bkg_estimator=MMMBackground()
    )
    
    # Set up photometry
    grouper = SourceGrouper(min_separation=2.0 * fwhm)
    
    photometry = PSFPhotometry(
        psf_model=psf_model,
        fit_shape=(11, 11),
        grouper=grouper,
        localbkg_estimator=bkg_estimator,
        aperture_radius=fwhm
    )
    
    # Perform photometry
    result = photometry(data=data, init_params=sources)
    
    print(f"PSF photometry complete for {len(result)} sources")
    return result


def write_results(output_path, matched_sources):
    """
    Write results to CSV file.
    
    Parameters
    ----------
    output_path : str or Path
        Output CSV file path
    matched_sources : astropy.table.Table
        Matched sources with photometry and APASS data
    """
    print(f"Writing results to: {output_path}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': matched_sources['source_id'],
        'flux': matched_sources['flux'],
        'Vmag': matched_sources['Vmag']
    })
    
    # Write to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully wrote {len(df)} sources to CSV")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Perform PSF photometry on FITS file with astrometry and APASS matching'
    )
    parser.add_argument('fits_file', type=str, help='Input FITS file')
    parser.add_argument('-o', '--output', type=str, default='photometry_results.csv',
                       help='Output CSV file (default: photometry_results.csv)')
    parser.add_argument('--fwhm', type=float, default=3.0,
                       help='FWHM of stars in pixels (default: 3.0)')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='Detection threshold in sigma (default: 5.0)')
    parser.add_argument('--tolerance', type=float, default=2.0,
                       help='Catalog matching tolerance in arcsec (default: 2.0)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Astrometry.net API key')
    
    args = parser.parse_args()
    
    try:
        # Read FITS file
        data, header = read_fits(args.fits_file)
        
        # Perform plate solving
        wcs = solve_plate(args.fits_file, api_key=args.api_key)
        
        # Detect sources
        sources = detect_sources(data, fwhm=args.fwhm, threshold=args.threshold)
        
        # Query APASS catalog
        apass_catalog = query_apass(wcs, data.shape)
        
        # Match catalogs
        matched = match_catalogs(sources, wcs, apass_catalog, tolerance=args.tolerance)
        
        # Perform PSF photometry (update flux values)
        psf_result = perform_psf_photometry(data, sources, fwhm=args.fwhm)
        
        # Create a mapping from source id to PSF flux using the 'id' column
        psf_flux_map = {psf_result['id'][i]: psf_result['flux_fit'][i] for i in range(len(psf_result))}
        
        # Update flux values in matched table with PSF photometry results
        for i, source_id in enumerate(matched['source_id']):
            if source_id in psf_flux_map:
                matched['flux'][i] = psf_flux_map[source_id]
        
        # Write results
        write_results(args.output, matched)
        
        print("\nProcessing complete!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

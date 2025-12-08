import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from pycountry import pycountry

from climada.entity import Exposures
from climada.util.constants import DEF_CRS
import climada.util.coordinates as u_coords

chunk_size = 1000000

def cropped_exposure_from_tifs(filepath_list, country, drop_zeros=True, band=1, src_crs=DEF_CRS, attrs={}, verbose=False):
    country_iso = pycountry.countries.get(name=country).alpha_3
    country_num = int(pycountry.countries.get(name=country).numeric)

    if not isinstance(filepath_list, list):
        filepath_list = [filepath_list]

    geom = u_coords.get_country_geometries(
        country_names = [country_iso],
        # extent = (0, 360, -90, 90),
        # center_crs = True
    ).geometry
    geom_gdf = gpd.GeoDataFrame(geometry=geom)
    geom_bounds = geom_gdf.total_bounds  # minx, miny, maxx, maxy    
    
    exp = Exposures.concat([
        _windowed_exposure_from_tif(
            file_path=filepath,
            geom_bounds=geom_bounds,
            band=band,
            source_crs=src_crs,
            verbose=verbose,
            attrs=attrs
        ) for filepath in filepath_list
    ])

    n_exp_before = exp.gdf.shape[0]
    if len(filepath_list) > 1:
        if verbose:
            print("Summing exposures at shared locations")
        new_gdf = exp.gdf.reset_index().groupby('geometry', as_index=False).agg('sum')
        exp.set_gdf(gpd.GeoDataFrame(new_gdf))
        n_exp_after = exp.gdf.shape[0]
        if verbose:
            print(f'... merged {n_exp_before - n_exp_after} duplicate points, {n_exp_after} points remain')
        n_exp_before = n_exp_after

    if verbose:
        print('Masking to country')

    n_chunks = int(np.ceil(n_exp_before / chunk_size))
    filtered_chunks = []
    countries_present = set()

    for i in range(n_chunks):
        if verbose and n_chunks > 1:
            print(f'... processing chunk {i+1} of {n_chunks}')
        chunk = exp.gdf.iloc[i*chunk_size:(i+1)*chunk_size].copy()
        chunk["region_id"] = u_coords.get_country_code(
            chunk.geometry.y,
            chunk.geometry.x,
            gridded=True
        )
        chunk_countries_present = set(chunk['region_id'].values)
        countries_present = countries_present.union(chunk_countries_present)
        chunk = chunk[chunk['region_id'] == country_num]
        filtered_chunks.append(chunk)

    exp.set_gdf(pd.concat(filtered_chunks, ignore_index=True))
    n_exp_after = exp.gdf.shape[0]
    if verbose:
        print(f'... removed {n_exp_before - n_exp_after} out of country points, {n_exp_after} points remain')
    n_exp_before = n_exp_after

    if drop_zeros:
        if verbose:
            print('Removing zero values')
        exp.set_gdf(exp.gdf[exp.gdf['value'] != 0])
        n_exp_after = exp.gdf.shape[0]
        if verbose:
            print(f'... removed {n_exp_before - n_exp_after} zero-value points, {n_exp_after} points remain')
        n_exp_before = n_exp_after
    
    return exp


def _windowed_exposure_from_tif(file_path, geom_bounds, band, source_crs, attrs, verbose):
    if verbose:
        print(f'Processing {file_path}')
        print("Creating rasterio Window")

    with rasterio.open(file_path) as src:
        win = from_bounds(geom_bounds[0], geom_bounds[1], geom_bounds[2], geom_bounds[3],
                          transform=src.transform)
        # expand window to integer offsets/lengths
        win = win.round_offsets().round_lengths()

    if verbose:
        print('Loading exposure data')

    if source_crs != 'EPSG:4326':
        print("Transforming raster data to WGS84 with rasterio default reprojection options")
        print("Warning: this might result in some loss of accuracy, I haven't check this yet")
    
    try:
        exp = Exposures.from_raster(
            file_name=file_path,
            band=band,
            src_crs=source_crs,
            dst_crs=DEF_CRS,
            window=win,
            attrs=attrs
        )
    except ValueError as e:
        if str(e) == "Input shapes do not overlap raster.":
            print('... no overlap between raster and country geometry, skipping')
            return Exposures()
        else:
            raise e
    
    return exp

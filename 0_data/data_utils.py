import geopandas as gpd
import rioxarray as rxr
from rasterstats import zonal_stats
import pandas as pd
import os
import rasterio
from rasterio.mask import mask
import numpy as np
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import pyreadr

def align_r1_to_r2(
    r1,
    r2,
    data_type="categorical"   # "categorical" or "continuous"
):
    """
    Align raster r1 to the grid of raster r2.

    Parameters
    ----------
    r1 : xarray.DataArray
        Source raster to be aligned.
    r2 : xarray.DataArray
        Target raster defining CRS, resolution, extent, and grid.
    data_type : str
        "categorical" or "continuous".
    clip : bool
        If True, crop r1 to r2 extent before resampling.

    Returns
    -------
    xarray.DataArray
        r1 aligned to r2 grid.
    """
    # 3. Choose resampling method
    if data_type == "nearest":
        resampling = Resampling.nearest
    elif data_type == "bilinear":
        resampling = Resampling.bilinear

    # 4. Resample onto r2 grid
    r1_aligned = r1.rio.reproject_match(
        r2,
        resampling=resampling
    )
    return r1_aligned, r2

def best_res_align(r1, r1catcon, r2, r2catcon,
                   region_bounds_buffered,
                   shape_crs, crop=True):
    """
    Align the raster with lower resolution to the raster with higher resolution.

    Parameters
    ----------
    r1 : xarray.DataArray
        First raster.
    r1catcon : str
        "categorical" or "continuous" for r1.
    r2 : xarray.DataArray
        Second raster.
    r2catcon : str
        "categorical" or "continuous" for r2.

    Returns
    -------
    aligned_r1 : xarray.DataArray
        r1 aligned to the higher-resolution raster grid (r2 or r1 swapped if needed).
    target_raster : xarray.DataArray
        The raster whose grid is being used as the reference.
    """

    # 1. CRS check / reproject if needed
    if r1.rio.crs != shape_crs:
        r1 = r1.rio.reproject(shape_crs)
    if r2.rio.crs != shape_crs:
        r2 = r2.rio.reproject(shape_crs)

    if crop:
        r1 = r1.squeeze().rio.clip_box(minx=region_bounds_buffered[0], miny=region_bounds_buffered[1],
                                                    maxx=region_bounds_buffered[2], maxy=region_bounds_buffered[3])
        r2 = r2.squeeze().rio.clip_box(minx=region_bounds_buffered[0], miny=region_bounds_buffered[1],
                                                    maxx=region_bounds_buffered[2], maxy=region_bounds_buffered[3])

    # 2. Compare resolutions
    res_r1_x, res_r1_y = r1.rio.resolution()
    # print('r1 resolution:', res_r1_x, res_r1_y)
    res_r2_x, res_r2_y = r2.rio.resolution()
    # print('r2 resolution:', res_r2_x, res_r2_y)

    # Use the smaller cell size as the target (higher-resolution raster)
    r1_avg_res = (abs(res_r1_x) + abs(res_r1_y)) / 2
    r2_avg_res = (abs(res_r2_x) + abs(res_r2_y)) / 2

    if r1_avg_res > r2_avg_res:
        # print('r1 is lower resolution, downsampling r1 to r2')
        # r1 is coarser -> align r1 to r2
        return align_r1_to_r2(r1, r2, data_type=r1catcon)
    else:
        # print('r2 is lower resolution, downsampling r2 to r1')
        # r2 is coarser -> align r2 to r1
        return align_r1_to_r2(r2, r1, data_type=r2catcon)[::-1]

def read_in(data_folder, admin, max_lag, start_year=2015, start_month=1, end_year=2024, end_month=12):
    valid_admin2 = pd.read_csv(os.path.join(data_folder, 'valid_admin/valid_admin2.csv'), header=None)[0].tolist()
    valid_admin1 = pd.read_csv(os.path.join(data_folder, 'valid_admin/valid_admin1.csv'), header=None)[0].tolist()
    valid_admin2.sort()
    valid_admin1.sort()
    va = {1: valid_admin1, 2: valid_admin2}

    # admin_year (urb, surv, urb_surv)
    admin_year_urbanisation = pd.read_csv(os.path.join(data_folder, f'admin_year_urbanisation/admin{admin}_year_urbanisation.csv'))
    admin_year_surveillance = pd.read_csv(os.path.join(data_folder, f'admin_year_surveillance/admin{admin}_year_surveillance.csv'))
    admin_year_urban_surveillance = pd.read_csv(os.path.join(data_folder, f'admin_year_urban_surveillance/admin{admin}_year_urban_surveillance.csv'))
    admin_year_pop = pd.read_csv(os.path.join(data_folder, f'admin_year_pop/admin{admin}_year_pop.csv'))

    admin_year = pd.merge(pd.merge(pd.merge(admin_year_urbanisation, admin_year_surveillance, on=[f'admin{admin}', 'year']),
                                   admin_year_urban_surveillance, on=[f'admin{admin}', 'year']),
                                   admin_year_pop, on=[f'admin{admin}', 'year'])
    new_cols = admin_year.columns.tolist()
    new_cols.remove('population')
    new_cols.insert(2, 'population')
    admin_year = admin_year.reindex(columns=new_cols)

    # admin_year_month (cases + ONI) + weather_stats(lag))

    # admin_year_month
    admin2_year_month_cases = d_incidence = pyreadr.read_r(os.path.join(data_folder, 'cases_deaths_pop_2016_2024_38.rds'))[None]
    admin2_year_month_cases = admin2_year_month_cases.loc[:, ['admin1', 'admin2', 'year', 'month', 'cases']]
    admin2_year_month_cases = admin2_year_month_cases.astype({'year': 'int32', 'month': 'int32'})
    admin_year_month_cases = (admin2_year_month_cases[admin2_year_month_cases[f"admin{admin}"].isin(va[admin])].sort_values([f"admin{admin}", 'year', 'month']).reset_index(drop=True))
    if admin==1:
        admin_year_month_cases = admin2_year_month_cases[['admin1', 'year', 'month', 'cases']].groupby(['admin1', 'year', 'month'], as_index=False).agg(lambda x: x.sum() if not x.isna().any() else np.nan)
        # size = admin2_year_month_cases[['admin1', 'year', 'month', 'cases']].groupby(['admin1', 'year', 'month'], as_index=False).size()
        # count = admin2_year_month_cases[['admin1', 'year', 'month', 'cases']].groupby(['admin1', 'year', 'month'], as_index=False).count()
    elif admin==2:
        admin_year_month_cases = admin2_year_month_cases
    else:
        raise ValueError("admin must be 1 or 2")
    
    # ONI
    ONI = pd.read_csv(os.path.join(data_folder, 'ONI/ONI.csv'))
    ONI = ONI.loc[(ONI['year'] >= 2016) & (ONI['year'] <= 2024), ['year', 'month', 'ONI']].reset_index(drop=True)

    admin_year_month = admin_year_month_cases.merge(ONI, on=['year', 'month'], how='left')

    # Climate statistics
    admin_year_month_climate_statistics = pd.read_csv(os.path.join(data_folder,
                                                                   f'admin_year_month_climate_statistics/admin{admin}_year_month_climate_statistics.csv'))
    for lag in range(0, max_lag + 1):
        admin_year_month_climate_statistics_lag = admin_year_month_climate_statistics.copy()
        new_cols = admin_year_month_climate_statistics_lag.columns[0:3].to_list() + [c+f' ({lag})' for c in admin_year_month_climate_statistics_lag.columns[3:]]
        admin_year_month_climate_statistics_lag.rename(columns={admin_year_month_climate_statistics_lag.columns[i]: new_cols[i] for i in range(len(new_cols))}, inplace=True)
        admin_year_month_climate_statistics_lag['month'] += lag
        admin_year_month_climate_statistics_lag.loc[admin_year_month_climate_statistics_lag['month'] > 12, 'year'] += 1
        admin_year_month_climate_statistics_lag.loc[admin_year_month_climate_statistics_lag['month'] > 12, 'month'] -= 12
        if lag==0:
            admin_year_month_climate_statistics_all_lags = admin_year_month_climate_statistics_lag
        else:
            admin_year_month_climate_statistics_all_lags = admin_year_month_climate_statistics_all_lags.merge(admin_year_month_climate_statistics_lag, on=[f'admin{admin}', 'year', 'month'], how='left')

    admin_year_month = admin_year_month.merge(admin_year_month_climate_statistics_all_lags, on=[f'admin{admin}', 'year', 'month'], how='left')
    full = admin_year_month.merge(admin_year, on=[f'admin{admin}', 'year'], how='left')

    full['date'] = pd.to_datetime(dict(year=full['year'], month=full['month'], day=1))
    mask1 = full['date'] >= pd.Timestamp(year=start_year, month=start_month, day=1)
    mask2 = full['date'] <= pd.Timestamp(year=end_year, month=end_month, day=1)
    full = full[mask1*mask2]
    full = full.drop(columns=['date'], inplace=False)
    return full
#!/usr/bin/env python

import numba
import xarray as xr
import numpy as np
from itertools import product #Cartesian product iterable

@numba.jit(nopython=True)
def calculate_interpolation(
    interp_template: np.ndarray,
    tropo_press: np.ndarray,
    mod_press: np.ndarray,
    averaging_kernel:np.ndarray,
    lenlatlon: list[tuple[int, int]],
    ) -> np.ndarray:
    """
    Calculate the interpolation. Logically, it should be part of interpolate_AK,
    but it was separated to take advantage of numba jit
    """
    interpolated: np.ndarray = interp_template
    for llat, llon in lenlatlon:
        tropo_press_mid: np.ndarray = (
            (tropo_press[llat, llon, 1:] + tropo_press[llat, llon, :-1]) /2
        )
        interpolated[:, llat, llon] = np.flip(
            np.interp(
                np.flip(mod_press[:,llat, llon]),
                np.flip(tropo_press_mid),
                np.flip(averaging_kernel[llat, llon, :])
            )
        )
    return interpolated
    


def interpolate_AK(ds_tropomi: xr.Dataset, ds_model: xr.Dataset) -> xr.DataArray:
    """
    Interpolate averaging kernel to
    model pressure layers
    """
    averaging_kernel: xr.DataArray = ds_tropomi["tropo_avg_kernel"].isel(time=0)
    tropo_press: xr.DataArray = ds_tropomi["hlevel_pressure"].isel(time=0)
    mod_press: np.ndarray = ds_model["pressure"].values
    lat: np.ndarray = ds_tropomi["latitude"].values
    lon: np.ndarray = ds_tropomi["longitude"].values

    interp_av: xr.DataArray = xr.zeros_like(ds_model["NO2_partialcolumn"]).rename(
        "tropo_avk"
    )
    lenlatlon = list(product(range(len(lat)), range(len(lon))))

    interp_av[:, :, :, 0] = calculate_interpolation(
        interp_template=interp_av[:,:,:,0].values,
        tropo_press=tropo_press.values,
        mod_press=mod_press,
        averaging_kernel=averaging_kernel.values,
        lenlatlon=lenlatlon
    )

    # for llat, llon in product(range(len(lat)), range(len(lon))):
    #     tropo_press_mid: xr.DataArray = (
    #         (tropo_press[llat, llon, 1:] + tropo_press[llat, llon, :-1]) /2
    #     )
    #     interp_av[:, llat, llon, 0] = np.flip(
    #         np.interp(
    #         np.flip(mod_press[:,llat,llon]),
    #         np.flip(tropo_press_mid),
    #         np.flip(averaging_kernel[llat, llon, :])
    #         )
    #     )

    interp_av.to_netcdf("interp_av.nc")
    return interp_av


def apply_avgk_no2(partial_col: xr.DataArray, avgk: xr.DataArray) -> xr.DataArray:
    """
    Apply TROPOMI's avg kernel, which should already be interpolated
    """
    output: xr.DataArray = xr.dot(avgk, partial_col, dims="lev")
    output.attrs = {
        "description": "Model NO2 after applying the averaging kernel",
        "units": "umol/m2"
    }
    return output


def merge_and_diff(no2_model: xr.DataArray, no2_tropomi: xr.DataArray) -> xr.Dataset:
    """
    Merge model and tropomi dataset. Use model lat and lon to
    account for tiny differences in the regridding process
    """
    ds_out: xr.Dataset = no2_model.to_dataset(name="model_no2")
    ds_out["tropomi_no2"] = xr.DataArray(
        data=no2_tropomi,
        dims=["latitude", "longitude"],
        coords={
            "latitude": ("latitude", no2_model["latitude"].values),
            "longitude": ("longitude", no2_model["longitude"].values)
        },
        attrs={
            "descrtiption": "Tropomi NO2 column",
            "units": "umol/m2"
        }
    )

    ds_out["difference"] = ds_out["model_no2"] - ds_out["tropomi_no2"]
    ds_out["difference"].attrs = {
        "description": "NO2 model column - NO2 tropomi column",
        "units": "umol/m2"
    }

    return ds_out


def main() -> None:
    ds_tropomi: xr.Dataset = xr.open_dataset(
        "KTW_S5P_NO2_06531_20190116_1710.nc4"
    )#.isel(time=0)
    ds_model: xr.Dataset = xr.open_dataset(
        "2MUSICAv0-SAM27_20190116_NO2_tropomi.nc"
    )#.isel(time=0)

    avgk: xr.DataArray = interpolate_AK(ds_tropomi, ds_model)
    avgk = avgk.where(avgk < 1e30).isel(time=0)
    partial_col: xr.DataArray = ds_model["NO2_partialcolumn"].isel(time=0)
    model_no2_col: xr.DataArray = apply_avgk_no2(partial_col, avgk)
    # ds_out: xr.Dataset = model_no2_col.to_dataset(name="model_no2")
    # ds_out["tropomi_no2"] = xr.DataArray(
    #     data=ds_tropomi["nitrogendioxide_tropospheric_column"].isel(time=0).values,
    #     dims = ["latitude", "longitude"],
    #     coords=dict(
    #         latitude=("latitude", ds_out["latitude"].values),
    #         longitude=("longitude", ds_out["longitude"].values)
    #     ),
    #     attrs=dict(
    #         description="NO2 retrieved by TROPOMI",
    #         units="umol/m2",
    #     )
    # )
    # ds_out["tropomi_no2"] = ds_out["tropomi_no2"].where(
    #     ds_out["tropomi_no2"] < 1e30
    # )
    # ds_out["difference"] = ds_out["model_no2"] - ds_out["tropomi_no2"]
    tropomi_no2_col = ds_tropomi["nitrogendioxide_tropospheric_column"].isel(time=0)
    ds_out: xr.Dataset = merge_and_diff(no2_model=model_no2_col,
                                        no2_tropomi=tropomi_no2_col)
    ds_out["perc_dif"] = ds_out["difference"] / ds_out["model_no2"] * 100
    ds_out.to_netcdf("comparison_20190116_06531.nc")


# ========================
if __name__ == "__main__":
    main()

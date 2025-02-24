import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        How to maximise IO performance by reading Zarr chunks concurrently?

        Relatively simple approach:

        1. Create a list of Zarr `chunks_to_load`. Probably do a month of NWP init times per batch. These chunks don't _have_ to perfectly align with the Zarr chunks. For example, for each task, load a geospatial rectangle around GB, without worrying whether that loads more than 1 Zarr chunk. But, for the other dimesnions, load a single chunk's worth (i.e. 1 `init_time`, all ens members, and all lead_times).
        2. `xarray` doesn't have an `async` API*. So use `ThreadPoolExecutor` (as per [the example](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example)) to load these chunks concurrently.
        3. `vstack` the data to the Polars dataframe in the main thread.

        \* alternatively, use Zarr-Python's `async` API to read chunks concurrently. But that is probably only worthwhile if we have to have hundreds of `GET` requests in flight in parallel; or if `xarray` complains about being called from multiple threads.
        """
    )
    return


@app.cell
def _():
    NWP_VARIABLES = [
        "downward_long_wave_radiation_flux_surface",
        "downward_short_wave_radiation_flux_surface",
        "percent_frozen_precipitation_surface",
        "precipitation_surface",
        "relative_humidity_2m",
        "temperature_2m",
        "total_cloud_cover_atmosphere",
        "wind_u_10m",
        "wind_v_10m",
        "wind_u_100m",
        "wind_v_100m",
    ]
    return (NWP_VARIABLES,)


@app.cell
def _():
    import pandas as pd
    import xarray as xr
    import altair as alt
    return alt, pd, xr


@app.cell
def _(xr):
    ds = xr.open_zarr(
        "https://data.dynamical.org/noaa/gefs/forecast/latest.zarr?email=jack@openclimatefix.org",
        decode_timedelta=True,
        # chunks=None,  # `chunks=None` disables Dask.
    )
    ds
    return (ds,)


@app.cell
def _(ds):
    """Select temperature data for GB."""

    _GB_LAT = slice(60, 49)
    _GB_LON = slice(-7, 2)
    temperature = ds["temperature_2m"].sel(
        init_time="2025-02-21T00",
        latitude=_GB_LAT,
        longitude=_GB_LON,
        # method="nearest",
    )
    return (temperature,)


@app.cell
def _(temperature):
    temperature.isel(ensemble_member=0, lead_time=0).plot()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

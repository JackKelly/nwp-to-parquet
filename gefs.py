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
    import pandas as pd
    import xarray as xr
    import altair as alt
    return alt, pd, xr


@app.cell
def _(xr):
    ds = xr.open_zarr(
        "https://data.dynamical.org/noaa/gefs/forecast/latest.zarr?email=jack@openclimatefix.org",
        decode_timedelta=True,
        chunks=None,  # `chunks=None` disables Dask.
    )
    ds
    return (ds,)


@app.cell
def _(ds):
    ds["temperature_2m"].chunks
    return


@app.cell
def _(ds, pd):
    """Select temperature data for London."""

    _LONDON_LAT = 51.51
    _LONDON_LON = -0.12
    temperature = (
        ds.sel(
            init_time="2025-02-21T00",
            latitude=_LONDON_LAT,
            longitude=_LONDON_LON,
            method="nearest",
        )["temperature_2m"]
        .to_dataframe()
        .reset_index()[["valid_time", "temperature_2m", "ensemble_member"]]
    )

    # Crop off the last few days of the forecast, because they're NaNs.
    _start_date = temperature["valid_time"].min()
    _end_date = _start_date + pd.Timedelta(days=22.5)
    temperature = temperature[temperature["valid_time"] <= _end_date]
    return (temperature,)


@app.cell
def _(alt, temperature):
    """Plot temperature data in an Altair Chart."""

    alt.Chart(temperature, title="GEFS Temperature Forecasts for London").mark_line(
        interpolate="monotone", opacity=0.3
    ).encode(
        alt.X("valid_time").axis(title="Date", format="%d %b"),
        alt.Y("temperature_2m").axis(title="°C", titleAngle=0, titleAlign="right"),
        detail="ensemble_member",
        color=alt.value("#707070"),
    ).properties(width=900, height=400).configure_axis(grid=False).configure_view(stroke=None)
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
    import zarr
    return (zarr,)


@app.cell
async def _(zarr):
    z = await zarr.api.asynchronous.open(
        store="https://data.dynamical.org/noaa/gefs/forecast/latest.zarr?email=jack@openclimatefix.org",
        mode="r",
    )
    z
    return (z,)


@app.cell
async def _(z):
    [a async for a in z.array_keys()]
    return


@app.cell
async def _(z):
    array = await z.get("temperature_2m")
    array
    return (array,)


@app.cell
def _(array):
    array.chunks
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

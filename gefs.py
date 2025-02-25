import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        How to maximise IO performance by reading Zarr chunks concurrently?

        Relatively simple approach:

        1. For each NWP variable: Use `dask` to load a month of data for one NWP variable, a geospatial rectangle around GB, all ens members, and all lead_times. Do any transforms necessary (e.g. get average for each GSP region; reduce to 8bit int)
        2. Save that month to parquet.
        3. Repeat for the next month!
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
    import numpy as np
    return alt, np, pd, xr


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
def _(ds, np):
    """Select temperature data for GB."""

    _GB_LAT = slice(60, 49)
    _GB_LON = slice(-7, 2)

    TEMPERATURE_MIN_C = -30
    TEMPERATURE_MAX_C = 50
    temperature_range_C = TEMPERATURE_MAX_C - TEMPERATURE_MIN_C

    temperature = (
        (
            (
                ds["temperature_2m"]
                .sel(
                    init_time=slice("2025-02-01T00", "2025-02-03T00"),
                    latitude=_GB_LAT,
                    longitude=_GB_LON,
                    # method="nearest",
                )
                .coarsen(latitude=2, longitude=2, boundary="trim")
                .mean()
                .clip(TEMPERATURE_MIN_C, TEMPERATURE_MAX_C)
                # TODO: Remove the fillna. Instead use Polar's ability to mark missing values.
                .fillna(TEMPERATURE_MIN_C)
                - TEMPERATURE_MIN_C
            )
            / (temperature_range_C / 255)
        )
        .round()
        .astype(np.uint8)
        .load()
    )

    print("   Number of pixels: {:4d}".format(len(temperature.latitude) * len(temperature.longitude)))
    print("Number of megabytes: {:6.1f} MB".format(temperature.nbytes / 1e6))
    return (
        TEMPERATURE_MAX_C,
        TEMPERATURE_MIN_C,
        temperature,
        temperature_range_C,
    )


@app.cell
def _(temperature):
    temperature.sel(init_time="2025-02-01T00").isel(ensemble_member=0, lead_time=0).plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Next steps:

        1. Save as Zarr v3 (with sharding?)
        2. Save as Parquet
        3. Compare size & read speed
        """
    )
    return


if __name__ == "__main__":
    app.run()

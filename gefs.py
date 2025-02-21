import marimo

__generated_with = "0.11.7"
app = marimo.App(width="medium")


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
    )
    ds
    return (ds,)


@app.cell
def _(ds, pd):
    """Select temperature data for London."""

    _LONDON_LAT = 51.51
    _LONDON_LON = -0.12
    temperature = (
        ds.sel(
            init_time="2025-02-21T00", latitude=_LONDON_LAT, longitude=_LONDON_LON, method="nearest"
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


if __name__ == "__main__":
    app.run()

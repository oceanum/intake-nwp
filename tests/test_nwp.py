from pathlib import Path
from datetime import datetime, timedelta
import intake
import pytest

from intake_nwp.utils import round_time
from intake_nwp.source.nwp import ForecastSource, NowcastSource


HERE = Path(__file__).parent


def test_forecast():
    source = ForecastSource(
        cycle="20231122T00",
        model="gfs",
        fxx=[0, 1, 2],
        priority=["google", "aws", "nomads", "azure"],
        product="pgrb2.0p25",
        pattern="ICEC",
        mapping={"longitude": "lon", "latitude": "lat", "siconc": "icecsfc"},
        sorted=True,
        max_threads="auto",
    )
    dset = source.to_dask()
    assert dset.time.size == 3
    assert "icecsfc" in dset


def test_forecast_latest():
    cycle_step = 12
    cycle = round_time(datetime.utcnow(), hour_resolution=cycle_step)
    source = ForecastSource(
        model="gfs",
        fxx=[0, 1, 2],
        priority=["google", "aws", "nomads", "azure"],
        product="pgrb2.0p25",
        pattern="ICEC",
        mapping={"longitude": "lon", "latitude": "lat", "siconc": "icecsfc"},
        sorted=True,
        cycle_step=cycle_step,
        max_threads="auto",
    )
    dset = source.to_dask()
    assert dset.time.to_index()[0] in (cycle, cycle - timedelta(hours=cycle_step))


def test_forecast_fxx_dict():
    source = ForecastSource(
        cycle="20231122T00",
        model="gfs",
        fxx={"start": 0, "stop": 3, "step": 1},
        priority=["google", "aws", "nomads", "azure"],
        product="pgrb2.0p25",
        pattern="ICEC",
        mapping={"longitude": "lon", "latitude": "lat", "siconc": "icecsfc"},
        sorted=True,
        max_threads="auto",
    )
    dset = source.to_dask()
    assert dset.time.size == 3
    assert "icecsfc" in dset


def test_nowcast():
    source = NowcastSource(
        start="20231101T00",
        stop="20231101T09",
        cycle_step=6,
        time_step=3,
        model="gfs",
        product="pgrb2.0p50",
        pattern="ICEC",
        priority=["google", "aws", "nomads", "azure"],
        mapping={"longitude": "lon", "latitude": "lat", "siconc": "icecsfc"},
        sorted=True,
        max_threads="auto",
    )
    dset = source.to_dask()
    assert dset.time.size == 4


@pytest.mark.parametrize("dataset_id", ["fc_gfs_icec", "nc_gfs_icec"])
def test_catalog(dataset_id):
    cat = intake.open_catalog(HERE / "catalog.yml")
    cat[dataset_id].to_dask()

from pathlib import Path
from datetime import datetime
import intake

from intake_nwp.utils import round_time
from intake_nwp.source.nwp import NWPSource


HERE = Path(__file__).parent


def test_nwp():
    source = NWPSource(
        cycle="20231122T00",
        model="gfs",
        fxx=[0, 1, 2],
        priority=["google", "aws", "nomads", "azure"],
        product="pgrb2.0p25",
        pattern="ICEC",
        mapping={"longitude": "lon", "latitude": "lat", "siconc": "icecsfc"},
        sorted=True,
    )
    dset = source.to_dask()
    assert dset.time.size == 3
    assert "icecsfc" in dset


def test_nwp_latest():
    cycle_step = 12
    cycle = round_time(datetime.utcnow(), hour_resolution=cycle_step)
    source = NWPSource(
        model="gfs",
        fxx=[0, 1, 2],
        priority=["google", "aws", "nomads", "azure"],
        product="pgrb2.0p25",
        pattern="ICEC",
        mapping={"longitude": "lon", "latitude": "lat", "siconc": "icecsfc"},
        sorted=True,
        cycle_step=cycle_step,
    )
    dset = source.to_dask()
    assert dset.time.to_index()[0] == cycle


def test_nwp_fxx_dict():
    source = NWPSource(
        cycle="20231122T00",
        model="gfs",
        fxx={"start": 0, "stop": 3, "step": 1},
        priority=["google", "aws", "nomads", "azure"],
        product="pgrb2.0p25",
        pattern="ICEC",
        mapping={"longitude": "lon", "latitude": "lat", "siconc": "icecsfc"},
        sorted=True,
    )
    dset = source.to_dask()
    assert dset.time.size == 3
    assert "icecsfc" in dset


def test_nwp_catalog():
    cat = intake.open_catalog(HERE / "catalog.yml")
    dset = cat.gfs_icec.to_dask()
    assert dset.time.size == 6

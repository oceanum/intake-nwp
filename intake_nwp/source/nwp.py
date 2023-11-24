import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Union

from intake_nwp.source import DataSourceMixin
from intake_nwp.utils import round_time


logger = logging.getLogger(__name__)


class NWPSource(DataSourceMixin):
    """Open a opendap datasource with netcdf4 driver

    Parameters
    ----------
    model: str
        Model type, e.g., {'hrrr', 'hrrrak', 'rap', 'gfs', 'ecmwf'}.
    fxx: Union[list[int], dict]
        Forecast lead time in hours, e.g., [0, 1, 2], dict(start=0, stop=3, step=1).        
    product: str
        Output variable product file type, e.g., {'sfc', 'prs', 'pgrb2.0p50'}.
    pattern: str
        Pattern to match the variable name in grib file to retain.
    cycle : Union[str, datetime, list[str], list[datetime]]
        Model initialisation cycle.
    cycle_step: int
        The interval between cycles in hours for retaining the latest cycle available.
    stepback: int
        The maximum number of cycles to step back to find the latest available cycle.
    priority: list[str]
        List of model sources to get the data in the order of download priority.
    mapping: dict
        Mapping to rename variables in the dataset.
    sorted: bool
        Sort the coordinates of the dataset.
    metadata: dict
        Extra metadata to add to data source.

    Notes
    -----
    * If fxx is a dict it is expected to have the keys 'start', 'stop', and 'step' to
      define the forecast lead time range from `numpy.arange`.
    * A ValueError exception is raised if the lead time defined by cycle and fxx is not
      entirely available.

    """

    name = "nwp"

    def __init__(
        self,
        model: str,
        fxx: Union[list[int], dict],
        product: str,
        pattern: str,
        cycle: Union[str, datetime] = None,
        cycle_step: int = 6,
        stepback: int = 0,
        priority: list[str] = ["google", "aws", "nomads", "azure"],
        mapping: dict = {},
        sorted: bool = False,
        metadata: dict = None,
        **kwargs
    ):
        super().__init__(metadata=metadata, **kwargs)

        self.model = model
        self.product = product
        self.pattern = pattern
        self.cycle = cycle
        self.cycle_step = cycle_step
        self.stepback = stepback
        self.priority = priority
        self.mapping = mapping
        self.sorted = sorted

        # Convert lead times to the expected format
        if isinstance(fxx, dict):
            fxx = [int(v) for v in np.arange(**fxx)]
        self.fxx = fxx

        # Set latest available cycle
        self._stepback = 0
        self._latest = round_time(datetime.utcnow(), hour_resolution=self.cycle_step)

        self._ds = None

    def __repr__(self):
        return (
            f"<NWPSource: cycle='{self.cycle}', model='{self.model}', fxx={self.fxx}, "
            f"product='{self.product}', pattern='{self.pattern}', "
            f"priority={self.priority}>"
        )

    def _set_latest_cycle(self):
        """Set cycle from the latest data available if cycle is not specified."""
        from herbie import Herbie

        if self.cycle:
            return self.cycle

        # Inspect data for latest cycle, step back if not found up to stepback limit
        f = Herbie(
            date=self._latest,
            model=self.model,
            fxx=0,
            product=self.product,
            priority=self.priority,
        )
        try:
            # Inventory raises a ValueError if no data can be found
            f.inventory(self.pattern)
            self.cycle = self._latest
        except ValueError:
            # Step back a cycle only if stepback limit is not reached
            if self._stepback >= self.stepback:
                raise ValueError(
                    f"No data found after {self.stepback} stepbacks for the given "
                    f"parameters: {self}"
                )
            self._stepback += 1
            self._latest -= timedelta(hours=self.cycle_step)
            return self._set_latest_cycle()

    def _open_dataset(self):
        from herbie import FastHerbie

        # Set latest cycle if not specified
        self._set_latest_cycle()

        fh = FastHerbie(
            [self.cycle],
            model=self.model,
            fxx=self.fxx,
            product=self.product,
            priority=self.priority,
        )
        for obj in fh.objects:
            logger.debug(obj)

        # Throw more meaningful error if no data found
        try:
            logger.debug(f"Inventory:\n{fh.inventory(self.pattern)}")
        except ValueError as e:
            raise ValueError(f"No data found for the given parameters: {self}") from e

        # Open the xarray dataset
        ds = fh.xarray(self.pattern, remove_grib=True)

        # Ensure single dataset is returned
        if isinstance(ds, list):
            raise ValueError(
                f"The given parameters: {self} returned multiple datasets that cannot "
                f"be concatenated, please review your selected pattern: {self.pattern}"
            )

        # Turn step index into time index
        ds = ds.assign_coords({"step": ds.valid_time}).drop(["time", "valid_time"])
        ds = ds.rename({"step": "time"}).reset_coords()

        # Ensure the entire lead time period requested is available
        times = ds.time.to_index()
        t1 = times[0] + timedelta(hours=self.fxx[-1])
        if t1 > times[-1]:
            raise ValueError(
                f"Data not available for the requested forecast lead time for {self}, "
                f"requested: {times[0]} - {t1}, available: {times[0]} - {times[-1]}"
            )

        # Sorting
        if self.sorted:
            for coord in ds.coords:
                ds = ds.sortby(coord)

        # Renaming
        ds = ds.rename(self.mapping)

        self._ds = ds

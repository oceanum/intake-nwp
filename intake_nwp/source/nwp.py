import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union

from intake_nwp.source import DataSourceMixin
from intake_nwp.utils import round_time


logger = logging.getLogger(__name__)


class NWPSource(DataSourceMixin):
    """Forecast data source.

    This driver opens a forecast dataset using the Herbies package.

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
        stepback: int = 1,
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

        self._fxx = fxx
        self._stepback = 0

        # Set latest available cycle
        self._latest = round_time(datetime.utcnow(), hour_resolution=self.cycle_step)

        self._ds = None

    def __repr__(self):
        return (
            f"<NWPSource: cycle='{self.cycle}', model='{self.model}', fxx={self.fxx}, "
            f"product='{self.product}', pattern='{self.pattern}', "
            f"priority={self.priority}>"
        )

    @property
    def fxx(self):
        """Convert lead times to the expected format."""
        if isinstance(self._fxx, dict):
            # Haven't figure out how to pass parameters as integers yet
            self._fxx = {k: int(v) for k, v in self._fxx.items()}
            self._fxx = [int(v) for v in np.arange(**self._fxx)]
        return self._fxx

    def _set_latest_cycle(self):
        """Set cycle from the latest data available if cycle is not specified."""
        from herbie import Herbie

        # Skip if cycle is specified
        if self.cycle:
            return self.cycle

        # Inspect data for latest cycle, step back if not found up to stepback limit
        f = Herbie(
            date=self._latest,
            model=self.model,
            fxx=self.fxx[-1],
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
        ds = fh.xarray(self.pattern)

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


class NowcastSource(DataSourceMixin):
    """Nowcast data source.

    This driver opens a nowcast dataset using the Herbies package.

    Parameters
    ----------
    model: str
        Model type, e.g., {'hrrr', 'hrrrak', 'rap', 'gfs', 'ecmwf'}.
    product: str
        Output variable product file type, e.g., {'sfc', 'prs', 'pgrb2.0p50'}.
    pattern: str
        Pattern to match the variable name in grib file to retain.
    start: Union[str, datetime]
        Start date of the nowcast.
    stop: Union[str, datetime]
        Stop date of the nowcast, by default the latest available cycle is used.
    cycle_step: int
        The interval between cycles in hours.
    time_step: int
        The interval between time steps in the nowcast in hours.
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

    """

    name = "nowcast"

    def __init__(
        self,
        model: str,
        product: str,
        pattern: str,
        start: Union[str, datetime],
        stop: Union[str, datetime] = None,
        cycle_step: int = 6,
        time_step: int = 1,
        stepback: int = 1,
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
        self.start = start
        self.stop = stop
        self.cycle_step = cycle_step
        self.time_step = time_step
        self.stepback = stepback
        self.priority = priority
        self.mapping = mapping
        self.sorted = sorted

        self._stepback = 0

        # Set latest available cycle
        self._latest = round_time(datetime.utcnow(), hour_resolution=self.cycle_step)

        self._ds = None

    def __repr__(self):
        return (
            f"<NWPSource: start='{self.start}', stop='{self.stop}', "
            f"cycle_step='{self.cycle_step}', time_step='{self.time_step}', "
            f"model='{self.model}', product='{self.product}', "
            f"pattern='{self.pattern}', priority={self.priority}>"
        )

    @property
    def DATES(self):
        """Dates of all cycles to use for nowcast."""
        dates = pd.date_range(
            start=self.start, end=self.stop, freq=f"{self.cycle_step}h"
        )
        return list(dates.to_pydatetime())

    @property
    def fxx(self):
        """Lead times to keep in each cycle."""
        if self.time_step > self.cycle_step:
            raise ValueError(
                f"Time step '{self.time_step}' must be less than or equal to cycle "
                f"step '{self.cycle_step}'"
            )
        if self.cycle_step % self.time_step != 0:
            raise ValueError(
                f"Cycle step '{self.cycle_step}' must be a multiple of time step "
                f"'{self.time_step}'"
            )
        return [int(v) for v in np.arange(0, self.cycle_step, self.time_step)]

    def _set_latest_cycle(self):
        """Set cycle from the latest data available if stop is not specified."""
        from herbie import Herbie

        # Skip if stop is already specified
        if self.stop:
            return self.stop

        # Inspect data for latest cycle, step back if not found up to stepback limit
        f = Herbie(
            date=self._latest,
            model=self.model,
            fxx=self.cycle_step,
            product=self.product,
            priority=self.priority,
        )
        try:
            # Inventory raises a ValueError if no data can be found
            f.inventory(self.pattern)
            self.stop = self._latest
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

    def _format_dataset(self, ds):
        """Format the dataset."""
        # Convert time and step indices into single time index.
        ds = ds.stack(times=["time", "step"], create_index=False)
        ds = ds.assign_coords({"times": ds.time + ds.step}).transpose("times", ...)
        ds = ds.drop(["time", "step", "valid_time"]).rename({"times": "time"})
        ds = ds.reset_coords()
        # Sorting
        if self.sorted:
            for coord in ds.coords:
                ds = ds.sortby(coord)
        # Renaming
        ds = ds.rename(self.mapping)
        return ds

    def _open_dataset(self):
        from herbie import FastHerbie

        # Set latest cycle if not specified
        self._set_latest_cycle()

        fh = FastHerbie(
            DATES=self.DATES,
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
        ds = fh.xarray(self.pattern)

        # Ensure single dataset is returned
        if isinstance(ds, list):
            raise ValueError(
                f"The given parameters: {self} returned multiple datasets that cannot "
                f"be concatenated, please review your selected pattern: {self.pattern}"
            )

        # Format dataset
        self._ds = self._format_dataset(ds)

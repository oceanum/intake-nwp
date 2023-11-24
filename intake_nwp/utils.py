from datetime import datetime


def round_time(time, hour_resolution=6) -> datetime:
    """Round time to the previous hour defined by hour_resolution.

    Parameters
    ----------
    hour_resolution: int
        The hour resolution to round to.

    Returns
    -------
    datetime
        The rounded time.

    """
    rounded_hour = (time.hour // hour_resolution) * hour_resolution
    rounded_time = time.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)
    return rounded_time

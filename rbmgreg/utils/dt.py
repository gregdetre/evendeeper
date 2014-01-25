from datetime import datetime, date, timedelta

def dt_str(dt=None, hoursmins=True, seconds=True):
    """
    Returns the current date/time as a yymmdd_HHMM_S string,
    e.g. 091016_1916_21 for 16th Oct, 2009, at 7.16pm in the
    evening.

    By default, returns for NOW, unless you feed in DT.
    """
    if dt is None:
        dt = datetime.utcnow()
    fmt = '%y%m%d'
    if hoursmins:
        fmt += '_%H%M'
    if seconds:
        fmt += '_%S'
    return dt.strftime(fmt)


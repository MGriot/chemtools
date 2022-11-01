from datetime import date, datetime


def when_date():
    """Function that give to you the date, express by year-mounth-day, of today

    Returns:
        string: today date in format %Y-%m-%d
    """
    now = datetime.now()
    return str(now.strftime("%Y-%m-%d"))

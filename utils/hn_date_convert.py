import pandas as pd

def hn_dt_convert(comments_hn):

    # Split the time_text column to get only the ISO 8601 string
    datetime_str = comments_hn['time_text'].str.split().str[0]

    # Convert to pandas datetime
    comments_hn['datetime_obj'] = pd.to_datetime(datetime_str)

    # Get the day of the week
    comments_hn['day_of_week'] = comments_hn['datetime_obj'].dt.day_name()
    comments_hn.drop(columns='time_text', inplace=True)

    return comments_hn
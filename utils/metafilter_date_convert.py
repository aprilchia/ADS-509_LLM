import pandas as pd
import re

def convert(main_df, comments_df):
    """Converts timestamp and date columns in comments_df to a proper datetime.

    Uses the year from main_df's date column and handles year rollover
    (e.g., post in Nov 2025, comment in Jan 2026).

    Args:
        main_df: DataFrame with 'thread_link' and 'date' columns (date as datetime)
        comments_df: DataFrame with 'thread_link', 'timestamp', and 'date' columns

    Returns:
        comments_df with new 'created_at' column and original timestamp/date removed
    """
    # 1. Merge to get the post date from main_df
    comments_with_year = comments_df.merge(
        main_df[['thread_link', 'date']].rename(columns={'date': 'post_date'}),
        on='thread_link'
    )

    # 2. Start with the post's year
    comments_with_year['year'] = comments_with_year['post_date'].dt.year

    # 3. Check if date already contains a year (4 consecutive digits)
    has_year = comments_with_year['date'].str.contains(r'\d{4}', regex=True)

    # 4. Build datetime string - only append year if not already present
    # Also strip any trailing comma from the date
    comments_with_year['date_clean'] = comments_with_year['date'].str.rstrip(',')

    comments_with_year['datetime_str'] = comments_with_year.apply(
        lambda row: f"{row['timestamp']} {row['date_clean']}"
        if re.search(r'\d{4}', str(row['date']))
        else f"{row['timestamp']} {row['date_clean']} {row['year']}",
        axis=1
    )

    # 5. Parse to datetime using mixed format to handle variations
    comments_with_year['created_at'] = pd.to_datetime(
        comments_with_year['datetime_str'],
        format='mixed'
    )

    # 6. Fix year rollover: if comment datetime is BEFORE post date, it must be next year
    rollover_mask = comments_with_year['created_at'] < comments_with_year['post_date']
    comments_with_year.loc[rollover_mask, 'created_at'] += pd.DateOffset(years=1)

    #7. Add day of week column
    comments_with_year["day_of_week"] = comments_with_year['created_at'].dt.day_name()

    # 7. Clean up temporary columns
    comments_with_year.drop(
        columns=['post_date', 'year', 'datetime_str', 'timestamp', 'date', 'date_clean'],
        inplace=True
    )

    return comments_with_year

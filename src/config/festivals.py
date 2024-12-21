"""Configuration file for festival data."""

from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

def get_chinese_new_year(year: int) -> Dict[str, str]:
    """Get Chinese New Year dates for a given year.
    Source: Hardcoded for accuracy as it follows lunar calendar
    """
    dates = {
        2030: ('02-17', '03-03'),
        2029: ('01-28', '02-11'),
        2028: ('02-10', '02-24'),
        2027: ('01-21', '02-04'),
        2026: ('02-01', '02-15'),
        2025: ('01-21', '02-04'),
        2024: ('02-10', '02-24'),
        2023: ('01-22', '02-05'),
        2022: ('02-01', '02-15'),
        2021: ('02-12', '02-26'),
        2020: ('01-25', '02-08'),
        2019: ('02-05', '02-19'),
        2018: ('02-16', '03-02'),
        2017: ('01-28', '02-11'),
        2016: ('02-08', '02-22'),
        2015: ('02-19', '03-05'),
        2014: ('01-31', '02-14'),
    }
    
    if year not in dates:
        return None
    
    start, end = dates[year]
    return {
        'name': f'Chinese New Year {year}',
        'start_date': f'{year}-{start}',
        'end_date': f'{year}-{end}'
    }

def get_christmas_dates(year: int) -> Dict[str, str]:
    """Get Christmas and New Year dates for a given year."""
    return {
        'name': f'Christmas {year}',
        'start_date': f'{year}-12-24',
        'end_date': f'{year+1}-01-02'
    }

def get_thanksgiving_dates(year: int) -> Dict[str, str]:
    """Get Thanksgiving dates for a given year (4th Thursday of November)."""
    # Find the first Thursday of November
    nov_first = datetime(year, 11, 1)
    days_until_thurs = (3 - nov_first.weekday()) % 7
    first_thurs = nov_first + timedelta(days=days_until_thurs)
    
    # Get the fourth Thursday
    thanksgiving = first_thurs + timedelta(weeks=3)
    end_date = thanksgiving + timedelta(days=3)  # Sunday after Thanksgiving
    
    return {
        'name': f'Thanksgiving {year}',
        'start_date': thanksgiving.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    }

def get_diwali_dates(year: int) -> Dict[str, str]:
    """Get Diwali dates for a given year.
    Source: Hardcoded for accuracy as it follows lunar calendar
    """
    dates = {
        2030: ('10-20', '10-24'),
        2029: ('11-01', '11-05'),
        2028: ('10-12', '10-16'),
        2027: ('11-01', '11-05'),
        2026: ('10-14', '10-18'),
        2025: ('10-24', '10-28'),
        2024: ('11-01', '11-05'),
        2023: ('11-12', '11-16'),
        2022: ('10-24', '10-28'),
        2021: ('11-04', '11-08'),
        2020: ('11-14', '11-18'),
        2019: ('10-27', '10-31'),
        2018: ('11-07', '11-11'),
        2017: ('10-19', '10-23'),
        2016: ('10-30', '11-03'),
        2015: ('11-11', '11-15'),
        2014: ('10-23', '10-27'),
    }
    
    if year not in dates:
        return None
    
    start, end = dates[year]
    return {
        'name': f'Diwali {year}',
        'start_date': f'{year}-{start}',
        'end_date': f'{year}-{end}'
    }

def get_ramadan_dates(year: int) -> Dict[str, str]:
    """Get Ramadan dates for a given year.
    Source: Hardcoded for accuracy as Hijri calendar calculations can be complex
    """
    dates = {
        2030: ('01-06', '02-04'),
        2029: ('01-17', '02-15'),
        2028: ('01-28', '02-26'),
        2027: ('02-08', '03-09'),
        2026: ('02-19', '03-20'),
        2025: ('03-01', '03-30'),
        2024: ('03-10', '04-09'),
        2023: ('03-22', '04-21'),
        2022: ('04-02', '05-02'),
        2021: ('04-13', '05-12'),
        2020: ('04-23', '05-23'),
        2019: ('05-05', '06-04'),
        2018: ('05-16', '06-14'),
        2017: ('05-26', '06-24'),
        2016: ('06-06', '07-05'),
        2015: ('06-17', '07-16'),
        2014: ('06-28', '07-27'),
    }
    
    if year not in dates:
        return None
    
    start, end = dates[year]
    return {
        'name': f'Ramadan {year}',
        'start_date': f'{year}-{start}',
        'end_date': f'{year}-{end}'
    }

def get_golden_week_dates(year: int) -> Dict[str, str]:
    """Get Golden Week dates for a given year (October 1-7)."""
    return {
        'name': f'Golden Week {year}',
        'start_date': f'{year}-10-01',
        'end_date': f'{year}-10-07'
    }

def generate_festival_dates(start_year: int = 2014, end_year: int = 2030) -> List[Dict[str, str]]:
    """Generate festival dates for the specified year range.
    
    Args:
        start_year: Start year (default: 2014)
        end_year: End year (default: 2030)
    
    Returns:
        List of dictionaries containing festival data
    """
    festivals = []
    
    for year in range(start_year, end_year + 1):
        # Chinese New Year
        cny = get_chinese_new_year(year)
        if cny:
            festivals.append({
                'name': cny['name'],
                'category': 'Chinese New Year',
                'start_date': cny['start_date'],
                'end_date': cny['end_date']
            })
        
        # Christmas and New Year
        christmas = get_christmas_dates(year)
        festivals.append({
            'name': christmas['name'],
            'category': 'Christmas and New Year',
            'start_date': christmas['start_date'],
            'end_date': christmas['end_date']
        })
        
        # Thanksgiving
        thanksgiving = get_thanksgiving_dates(year)
        festivals.append({
            'name': thanksgiving['name'],
            'category': 'Thanksgiving',
            'start_date': thanksgiving['start_date'],
            'end_date': thanksgiving['end_date']
        })
        
        # Diwali
        diwali = get_diwali_dates(year)
        if diwali:
            festivals.append({
                'name': diwali['name'],
                'category': 'Diwali',
                'start_date': diwali['start_date'],
                'end_date': diwali['end_date']
            })
        
        # Ramadan
        ramadan = get_ramadan_dates(year)
        if ramadan:
            festivals.append({
                'name': ramadan['name'],
                'category': 'Ramadan',
                'start_date': ramadan['start_date'],
                'end_date': ramadan['end_date']
            })
        
        # Golden Week
        golden_week = get_golden_week_dates(year)
        festivals.append({
            'name': golden_week['name'],
            'category': 'Golden Week',
            'start_date': golden_week['start_date'],
            'end_date': golden_week['end_date']
        })
    
    return festivals

# Generate festival dates from 2014 to 2030
FESTIVALS = generate_festival_dates() 
"""Configuration for major market events that impacted Bitcoin price."""

from datetime import datetime

MAJOR_EVENTS = [
    {
        'date': datetime(2022, 11, 8),
        'event': 'FTX Collapse',
        'description': 'FTX cryptocurrency exchange collapse',
        'type': 'negative',
        'impact': 'Major market crash and loss of confidence'
    },
    {
        'date': datetime(2022, 5, 7),
        'event': 'LUNA/UST Crash',
        'description': 'Terra/LUNA ecosystem collapse',
        'type': 'negative',
        'impact': 'Triggered broader crypto market selloff'
    },
    {
        'date': datetime(2022, 6, 12),
        'event': 'stETH Depeg',
        'description': 'Lido stETH depegged from ETH',
        'type': 'negative',
        'impact': 'Caused liquidity crisis in DeFi'
    },
    {
        'date': datetime(2016, 11, 8),
        'event': 'Trump Election',
        'description': 'Donald Trump elected as US President',
        'type': 'neutral',
        'impact': 'Increased market uncertainty'
    },
    {
        'date': datetime(2024, 1, 10),
        'event': 'Spot Bitcoin ETF',
        'description': 'SEC approves spot Bitcoin ETFs',
        'type': 'positive',
        'impact': 'Major institutional adoption milestone'
    },
    {
        'date': datetime(2024, 1, 10),
        'event': 'Grayscale ETF',
        'description': 'Grayscale wins Bitcoin ETF approval',
        'type': 'positive',
        'impact': 'GBTC conversion to ETF approved'
    },
    {
        'date': datetime(2021, 4, 14),
        'event': 'Coinbase IPO',
        'description': 'Coinbase goes public on NASDAQ',
        'type': 'positive',
        'impact': 'First major crypto exchange IPO'
    },
    {
        'date': datetime(2021, 9, 7),
        'event': 'El Salvador Bitcoin',
        'description': 'El Salvador adopts Bitcoin as legal tender',
        'type': 'positive',
        'impact': 'First country to adopt Bitcoin as legal tender'
    },
    {
        'date': datetime(2021, 2, 8),
        'event': 'Tesla Bitcoin',
        'description': 'Tesla buys $1.5B in Bitcoin',
        'type': 'positive',
        'impact': 'Major corporate adoption'
    },
    {
        'date': datetime(2020, 3, 12),
        'event': 'COVID Crash',
        'description': 'Market crash due to COVID-19 pandemic',
        'type': 'negative',
        'impact': 'Global market panic'
    },
    {
        'date': datetime(2023, 3, 10),
        'event': 'SVB Collapse',
        'description': 'Silicon Valley Bank collapse',
        'type': 'negative',
        'impact': 'Banking crisis affects crypto markets'
    }
]

HALVING_EVENTS = [
    {
        'date': datetime(2012, 11, 28),
        'event': 'First Halving',
        'description': 'Block reward reduced from 50 to 25 BTC',
        'block_height': 210000
    },
    {
        'date': datetime(2016, 7, 9),
        'event': 'Second Halving',
        'description': 'Block reward reduced from 25 to 12.5 BTC',
        'block_height': 420000
    },
    {
        'date': datetime(2020, 5, 11),
        'event': 'Third Halving',
        'description': 'Block reward reduced from 12.5 to 6.25 BTC',
        'block_height': 630000
    },
    {
        'date': datetime(2024, 4, 20),
        'event': 'Fourth Halving',
        'description': 'Block reward to reduce from 6.25 to 3.125 BTC',
        'block_height': 840000
    },
    {
        'date': datetime(2028, 5, 1),
        'event': 'Fifth Halving',
        'description': 'Block reward to reduce from 3.125 to 1.5625 BTC',
        'block_height': 1050000
    }
] 
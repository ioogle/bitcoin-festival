"""Configuration for major market events that impacted Bitcoin price."""

from datetime import datetime

MAJOR_EVENTS = [
    {
        'date': datetime(2024, 12, 2),
        'event': 'Trump 2nd Election',
        'description': 'Donald Trump elected as US President again',
        'type': 'positive',
        'impact': 'Trump is back in the White House'
    },
    {
        'date': datetime(2024, 1, 31),
        'event': 'Fed Rate Hold',
        'description': 'Fed maintains interest rates at 5.25-5.50%',
        'type': 'neutral',
        'impact': 'Fed signals potential rate cuts in 2024'
    },
    {
        'date': datetime(2023, 7, 26),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 5.25-5.50%',
        'type': 'negative',
        'impact': 'Last rate hike of 2023 tightening cycle'
    },
    {
        'date': datetime(2023, 5, 3),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 5.00-5.25%',
        'type': 'negative',
        'impact': 'Tenth consecutive rate increase'
    },
    {
        'date': datetime(2023, 3, 22),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 4.75-5.00%',
        'type': 'negative',
        'impact': 'Rate hike despite banking crisis'
    },
    {
        'date': datetime(2022, 12, 14),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 4.25-4.50%',
        'type': 'negative',
        'impact': 'Seventh rate hike of 2022'
    },
    {
        'date': datetime(2022, 11, 8),
        'event': 'FTX Collapse',
        'description': 'FTX cryptocurrency exchange collapse',
        'type': 'negative',
        'impact': 'Major market crash and loss of confidence'
    },
    {
        'date': datetime(2022, 9, 21),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 3.00-3.25%',
        'type': 'negative',
        'impact': 'Third consecutive 75 basis point hike'
    },
    {
        'date': datetime(2022, 7, 27),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 2.25-2.50%',
        'type': 'negative',
        'impact': 'Second consecutive 75 basis point hike'
    },
    {
        'date': datetime(2022, 6, 15),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 1.50-1.75%',
        'type': 'negative',
        'impact': 'Largest rate hike since 1994'
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
        'date': datetime(2022, 5, 4),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 0.75-1.00%',
        'type': 'negative',
        'impact': 'Largest rate hike since 2000'
    },
    {
        'date': datetime(2022, 3, 16),
        'event': 'Fed Rate Hike',
        'description': 'Fed raises rates to 0.25-0.50%',
        'type': 'negative',
        'impact': 'First rate hike since 2018'
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
        'date': datetime(2020, 3, 15),
        'event': 'Fed Emergency Cut',
        'description': 'Fed cuts rates to 0-0.25% due to COVID-19',
        'type': 'neutral',
        'impact': 'Emergency rate cut to near zero'
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
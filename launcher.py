import pathlib

import orchestrator


def main():
    pathlib.Path('/models').mkdir(exist_ok=True)
    pathlib.Path('/scalers').mkdir(exist_ok=True)
    orchestrator.orchestrate(symbols, True)

symbols = [
    'ADBE',
    'AMBA',
    'AAPL',
    'AMD',
    'AMZN',
    'ATVI',
    'BABA',
    'BIDU',
    'CEF',
    'CLX',
    'CMG',
    'CTXS',
    'DOCU',
    'DQ',
    'EA',
    'ENPH',
    'ERIC',
    'ESPO',
    'FB',
    'FCX',
    'GDX',
    'GOOG',
    'INTC',
    'ISRG',
    'LOGI',
    'LULU',
    'LVGO',
    'MSFT',
    'MTA',
    'MU',
    'NEM',
    'NETE',
    'NFLX',
    'NIO',
    'NOW',
    'OLED',
    'PAGS',
    'PINS',
    'PROS',
    'PYPL',
    'QDEL',
    'SEDG',
    'SHOP',
    'SONO',
    'SQ',
    'TCEHY',
    'TDOC',
    'TSLA',
    'TSM',
    'TTD',
    'TXN',
    'Z',
    'ZNGA']

if __name__ == "__main__":
    main()
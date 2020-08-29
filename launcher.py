import pathlib
from datetime import date, timedelta
import numpy as np
from stockatroncore import StockatronCore


def main():
    pathlib.Path('models').mkdir(exist_ok=True)
    pathlib.Path('runs').mkdir(exist_ok=True)
    np.random.seed(1)

    # symbols = ['APTV', 'BILI', 'BIDU', 'IKA.L', 'GOOG', 'ERIC', 'TM', 'LULU', 'PICK', 'NIO', 'PYPL', 'SQ'])

    core = StockatronCore(start_date=date.today() - timedelta(days=20 * 365))

    do_training = False
    for stock in ['NIO', 'PYPL', 'AMBA']:
        if do_training:
            core.train_model(symbol=stock)

        core.make_predictions(stock)


if __name__ == "__main__":
    main()
from datetime import date, timedelta
import numpy as np
from stockatroncore import StockatronCore


def main():
    np.random.seed(1)

    # symbols = ['APTV', 'BILI', 'BIDU', 'IKA.L', 'GOOG', 'ERIC', 'TM', 'LULU', 'PICK', 'NIO', 'PYPL', 'SQ'])

    core = StockatronCore(start_date=date.today() - timedelta(days=20 * 365))
    do_training = True
    for stock in ['NVAX', 'NIO', 'LULU', 'ERIC']:
        if do_training:
            core.train_model(symbol=stock)
        #core.make_predictions(stock)
        #core.analyse_latest_model(stock)

if __name__ == "__main__":
    main()
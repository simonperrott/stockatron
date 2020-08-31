import os
from datetime import date, datetime
import pandas as pd
from dtos import ModelContainer
import financer as yf


class StockatronLogger:

    def __init__(self, data_chef):
        self.data_chef = data_chef
        self.column_names = ['Symbol', 'Run_Date', 'Model', 'Model_Test_Score', 'Prediction', 'Actual']
        self.runs_log = os.path.join('runs', 'stockatron_runs.csv')

    def record_run(self, symbol, model_details:ModelContainer, prediction):
        data = {'Symbol': symbol, 'Run_Date': date.today().strftime("%Y-%m-%d"), 'Model': model_details.version, 'Model_Test_Score': model_details.test_score, 'Prediction': prediction}
        if os.path.exists(self.runs_log):
            date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
            df = pd.read_csv(self.runs_log, names=self.column_names, parse_dates=['Run_Date'], date_parser=date_parser)
            df.append(data, ignore_index=True)
        else:
            df = pd.DataFrame(data, index=[0], columns=self.column_names)
        #self.__update_record_with_actual(symbol, df)
        df.to_csv(self.runs_log)


    def __update_record_with_actual(self, symbol, df):
        df_predict = df[(df['Symbol'] == symbol) & (df['Actual'].isnull()) ]
        start_date = df_predict['Run Date'].min()
        df_actual = yf.get_ticker(symbol, start_date=start_date)
        df_actual['change'] = self.data_chef.calculate_price_change_ts(df_actual)
        df_actual['label'] = self.data_chef.create_labels(df_actual['change'])
        #for r in df_predict:
            # TODO: how to find the correct date that is 20 trading days ahead of prediction?
            # r['Actual'] = df_actual.loc[df_actual[''], 'label']

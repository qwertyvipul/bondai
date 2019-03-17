'''
https://query1.finance.yahoo.com/v7/finance/download/IBM?period1=0&period2=1550943181&interval=1d&events=history&crumb=7nzdw7DkuGR
'''

import get_yahoo_quotes
import os

class YahooFinanceDownloader:
    def __init__(self):
        self.data_folder = "data/"

        self.file_paths = {
            'data': 'yahoo-finance/'
        }

        self.file_names = {
            'historical-prices': 'historical-prices.csv'
        }

        self.url_paths = {
            'company': 'https://query1.finance.yahoo.com/v7/finance/download/',
            'historical-prices': {
                'part-a': '?period1=',
                'part-b': '&period2=',
                'part-c': '&interval=1d&events=history&crumb=7nzdw7DkuGR'
            }
        }

        self.tickers = []

    def downloadHistoricalPrices(self):
        for ticker in self.tickers:
            for key, file_name in self.file_names.items():
                # url = (self.url_paths['company'] + ticker + 
                #     self.url_paths['historical-prices']['part-a'] + "0" + 
                #     self.url_paths['historical-prices']['part-b'] + str(int(time.time())) +
                #     self.url_paths['historical-prices']['part-c']
                # )

                # print(url)

                data = get_yahoo_quotes.download_quotes(ticker)

                file_path = self.data_folder + self.file_paths['data'] + ticker
                # download = requests.get(url)
                # decoded_content = download.content.decode('utf-8')
                # cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                # my_list = list(cr)
                # for row in my_list:
                #     print(row)

                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                if(key == 'historical-prices'):
                    file_name = ticker + ".csv"
                output = open(file_path + "/" + file_name, 'wb')
                output.write(data.content)
                output.close()

yfd = YahooFinanceDownloader()
yfd.tickers = ["AAPL", "IBM", "SPY"]
yfd.downloadHistoricalPrices()
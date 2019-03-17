import requests
import os

class StockrowDownloader:
    def __init__(self):
        self.data_folder = ''

        self.file_paths = {
            'data': 'stockrow',
            'financials': '/financials',
            'annual': '/financials/annual',
            'quaterly': 'financials/quaterly'
        }

        self.file_names = {
            'income-statement': 'income-statement.xlsx',
            'balance-sheet': 'balance-sheet.xlsx',
            'cashflow-statement': 'cashflow-statement.xlsx',
            'metrics': 'metrics.xlsx',
            'growth': 'growth.xlsx'
        }

        self.url_paths = {
            'company': 'https://stockrow.com/api/companies/',
            'annual': {
                'income-statement': '/financials.xlsx?dimension=MRY&section=Income%20Statement&sort=desc',
                'balance-sheet': '/financials.xlsx?dimension=MRY&section=Balance%20Sheet&sort=desc',
                'cashflow-statement': '/financials.xlsx?dimension=MRY&section=Cash%20Flow&sort=desc',
                'metrics': '/financials.xlsx?dimension=MRY&section=Metrics&sort=desc',
                'growth': '/financials.xlsx?dimension=MRY&section=Growth&sort=desc'
            } 
        }

        self.tickers = []

    def download(self):
        for ticker in self.tickers:
            for key, file_name in self.file_names.items():
                url = self.url_paths['company'] + ticker + self.url_paths['annual'][key]
                file_path = self.data_folder + self.file_paths['data'] + "/" + ticker + self.file_paths['annual']
                file = requests.get(url)
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                output = open(file_path + "/" + file_name, 'wb')
                output.write(file.content)
                output.close()

            

# import requests
# dls = "https://stockrow.com/api/companies/AAPL/financials.xlsx?dimension=MRY&section=Income%20Statement&sort=desc"
# resp = requests.get(dls)

# output = open('income-statement.xlsx', 'wb')
# output.write(resp.content)
# output.close()

downloader = StockrowDownloader()
downloader.tickers = ["AAPL", "IBM"]
downloader.download()
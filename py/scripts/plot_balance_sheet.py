import pandas as pd
import matplotlib.pyplot as plt

bs_wb = pd.read_excel("../../data/stockrow/AAPL/financials/annual/balance-sheet.xlsx")
bs_data = bs_wb.as_matrix()
bs_cash = bs_data[0]

plt.plot(bs_cash[::-1])
plt.plot(bs_cash[::-1], 'ro')
plt.show()
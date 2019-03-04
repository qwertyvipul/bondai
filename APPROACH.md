### Approach

Our approach to develop this AI system would be as follows –

    1.	Generate Financial Dataset using APIs and web scrappers.

    2.	Generate supervised clusters using labelled financial sectors and product category of the company.

    3.	Generate up-supervised clusters based on the company’s financials.

    4.	Generate co-relation between the companies in supervised sectors.

    5.	Model co-relation between the financial numbers in the un-supervised sectors and sum up these co-relations into a 10-point scale credit rating neural network.

    6.	Train supervised regression models to predict the future performance of the company based on its past track record and the current financial health.

    7.	Plot the risk of prediction inaccuracy, uncertainty and default risks in the form of a risk timeline into the future.

    8.	Generate Event Dataset using APIs and financial news archives and SEC filings.

    9.	Train an NLP based model to predict the effect of events on the credit rating of the company.

    10.	Use this model to keep track of global financial events and update the credit ratings and risk timeline in real time.

    11.	Generate a report of the analysis and assumptions behind credit ratings and the risk timeline using NLP and pre-formatted templates.

    12.	Create a website connected to the AI system by a REST API to publish live credit ratings, risk timelines and corresponding report.

Note – By financials of a company we refer to the 10Q and 10K filings of the company and we want to keep our analysis models independent from public sentiments as well and therefore do not plan to use the stock prices of the company at any stage of development. 
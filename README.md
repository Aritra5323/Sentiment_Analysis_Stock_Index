# Sentiment Analysis of the Stock Index

The project aims to use various tools to analyse the trend of the stock index in accordance to the news published in the media.LDA and ARIMA models were used for the prediction of the sttock index.


## Authors

- [@Aritra5323](https://www.github.com/Aritra5323)


## Modules used
```bash
import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
```
Detailed codes and Outputs are given in the Sentiment Analysis Stocks.ipynb in the repository
## Features

- Easy Implementation
- Upgradation of the models
- User Friendly
- Usage of Methodologies can be altered


## FAQ

#### How can the above analysis be helpful for a particular individual or a company?

The stock index gives an overall view of how the market is performing. Whether the company/individual should start investing on certain stocks depends on the trend of the graph. Any news which hampers the market trend can be detrimental for the company/individual. Therefore, if the rise or fall of the index can be predicted using only the news headlines it would be beneficial for the company to gain larger profits or incur less losses by buying growing stocks or selling it at the right time. In the above workings, the Sentiment Analysis was performed which gave us the information whether the market would go up or down. Similarly fresh news headlines can be put in the model to give a fair idea of the situation. To get a more factual analysis the use of ARIMA model is established. It gives near accurate values of the actual stock index and also helps in predicting the future values of the index. Hence it is required to use both the Sentiment Analyser as well as the ARIMA model so that companies can plan whether to invest more or invest less according to the current situation.

Stock Analysis Data Science Project
===================
This is a data science exploration project that performs analysis on stocks of a
variety of tech companies. Using life data from Yahoo and Google, various
aspects of stock information are visualized, risks are analyzed, and finally
predictions are given for future stock prices using the Monte Carlo method.


----------

Data Requesting and Description
-------------
The Python data manipulation library *pandas* was used to acquire life stock data fromYahoo Finance. Stock data for a one-year period (from current date to a year before) for Apple, Google, Microsoft, and Amazon are retrieved. For each
company’s stock, the retrieved information includes the opening price, the
highest and the lowest prices, closing price, volume, and the adjusted closing
price. 


----------


Observations and Analysis
-------------------
Apple stock data was plotted for visualization, and the following observations was made: 
**Observations**:
1. Apple's closing prices experienced a significant decline in price starting from November 2018 and ending in January 2019. The price has since rebounded to about half of its previous peak, but is starting to decline again. 
2.  The volume of Apple stock spiked dramatically around the same time that it hit its lowest point in price.
3.  The stock's moving average followed the same trend as its price.
4.  Daily return was most variant while the stock was declining, then stabled out as it reached its lowest point and started rebounding. 

The analysis continued onto comparing the daily returns of each company
against each other. Interestingly enough, all the companies' daily returns seemed relatively correlated to each other. The most tightly related were between
Amazon and Microsoft, and Amazon and Apple. 



----------


Analyzing Risk
-------------
The "bootstrap" method, which entails calculating empirical quantiles from a histogram of daily returns, was used to analyze risk. After plotting the histogram, calculating the 0.05 empirical quantile of daily returns came out to -0.03123. This means that with 95% confidence, the worst daily loss would not exceed 3.123%. 

The Monte Carlo method to run many trials with random market conditions, and then calculate portfolio losses for each trial. After analyzing all the given variables, it was time to create a model based on what was learned. 

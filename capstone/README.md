# Machine Learning Engineer Nanodegree
## Capstone https://github.com/gilakos/machine-learning/tree/master/projects/capstone
Gil Akos 
April 7th, 2017

## Deep Learning Stock Value Predictor

### Project Overview

Investment firms, hedge funds, and automated trading systems have used programming and advanced modeling to interact with and profit from the stock market since computerization of the exchanges in the 1970s<sup>1</sup>.  Whether by means of better analysis, signal identification, or automating the frequency of trades, the goal has been to leverage technology in order create investment systems that outperform alternatives - either  service providers (competitors like alternative hedge funds) or products/benchmarks (ETFs or the S&P 500). 

Today, the most promising and adcendant technology, Deep Learning, is the target of incorporation into advanced investment systems<sup>2</sup> offered by "Artificially Intelligent Hedge Funds"<sup>3</sup> and "Deep Investing"<sup>4</sup> as-a-service startups, with claims of outperformance of the S&P 500 Index of up to 87%<sup>4</sup>. Given profit opportunity that large, can a basic Deep Learning model built with publicly available technology achieve positive predictive performance? Even an order of magnitude less of an advantage (8.7%) over the noise in the  market and a baseline of the S&P ETF (SPY) could be valuable for an amateur investor!

This project seeks to utilize Deep Learning models, specifically Recurrant Neural Nets (RNNs), to predict stock prices. Much academic work has been developed using this technique<sup>5</sup>, as well as similar studies using Boltzmann machines<sup>6</sup> for both momentum trading strategies and time series prediction. As discussed above and in the below articles from sources ranging from technology magazines (Wired<sup>3</sup>) to the standard bearer for market information (Financial Times<sup>2</sup>), these models are also being applied to real world trading platforms<sup>7</sup>. In this study, I will use Keras<sup>8</sup> to build a RNN to predict stock prices using historical closing price and trading volume and visualize both the predicted price values over time and the optimal parameters for the model.

### Project Organization
Project assets are organized as follows:

Project Submissions
- Proposal
- Report

Project Development
- Development (python notebooks)
- Examples

Project Assets (exports per study)
- HTMLs
- Graphs
- Diagrams

### Sequence of Project Studies
- Study 00 // Data Exploration // Gathering and Analyzing Stock Data
- Study 01 // Benchmark Model // Linear Regression with Sci-kit Learn
- Study 02 // Benchmark Model // Polynomial Regression with Sci-kit Learn
- Study 03 // Deep Learning Model // Multi-Layer Perceptron with Keras and Tensorflow
- Study 04 // Deep Learning Model // Long Short-term Memory “LSTM” with Keras and Tensorflow
- Study 05 // Deep Learning Model // LSTM with Variable “Lookback” Dataset (Adjusted Close)
- Study 06 // Deep Learning Model // LSTM with Variable “Lookback” Dataset (Trading Volume)
- Study 07 // Deep Learning Model // LSTM with two “Stacked” LSTM Layers (Adjusted Close)
- Study 08 // Deep Learning Model // LSTM with two “Merged” LSTM Branches (Adjusted Close + Volume)
- Study 09 // Deep Learning Model // LSTM with two “Merged” LSTM Branches with “Zero” Padding Datasets (Adjusted Close + Trading Volume)
- Study 10 // Deep Learning Model // LSTM with two “Merged” LSTM Branches with “i” Padding Datasets (Adjusted Close + Trading Volume)
- Study 11 // Deep Learning Model // LSTM with two “Merged” LSTM Branches (Adjusted Close + SPY Adjusted Close)
- Study 12 // Deep Learning Model // Final LSTM with Tuned Dataset (Adjusted Close) and Parameters
- Study 13 // Deep Learning Model // Final LSTM with Tuned Dataset (Adjusted Close) and Parameters using four previously unused tickers

-----------

### Footnotes
1. [Bloomberg // History of Algorithmic Trading Shows Promise and Perils](https://www.bloomberg.com/view/articles/2012-08-08/history-of-algorithmic-trading-shows-promise-and-perils)
2. [Financial Times // Money managers seek AI’s ‘deep learning’](https://www.ft.com/content/9278d1b6-1e02-11e6-b286-cddde55ca122)
3. [Wired // The Rise of the Artificially Intelligent Hedge Fund](https://www.wired.com/2016/01/the-rise-of-the-artificially-intelligent-hedge-fund/)
4. [Stocks Neural Net // About](https://stocksneural.net/about)
5. [Deep Learning for Time Series Modeling](http://cs229.stanford.edu/proj2012/BussetiOsbandWong-DeepLearningForTimeSeriesModeling.pdf)
6. [Applying Deep Learning to Enhance Momentum Trading Strategies in Stocks](http://cs229.stanford.edu/proj2013/TakeuchiLee-ApplyingDeepLearningToEnhanceMomentumTradingStrategiesInStocks.pdf)
7. [MIT Technology Review // Will AI-Powered Hedge Funds Outsmart the Market?](https://www.technologyreview.com/s/600695/will-ai-powered-hedge-funds-outsmart-the-market/)
8. [Keras // Deep Learning library for Theano and TensorFlow](https://keras.io)

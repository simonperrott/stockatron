# Stockatron Overview
Stockatron uses Keras to train a LSTM (Long Short-Term Memory) Recurrent Neural Network for any stock then uses it to predict how the stock's price will change over the coming days.

It's objective is to solve a multi-class classification challenge of whether the stock's price will increase by more than a specified %, decrease by more than a specified % or stay unchanged within these boundaries over the next specified period. It has initially been configured to classify % movements by the end of the next 5 trading days; a price increase > 10% as +1, a price decrease < 5% as -1 and a price change within these boundaries as 0.

The important components are:

## Stockatron Core

The Orchestrator & 'central nervous system' of Stockatron. It supports:
- Training of new models by:
  - Using a DataChef to prepare a timeseries of a stock's price changes for training a model.
  - Configuring model hyperparameters and calling a Trainer to create a model for the data and hyperparameters. 
  - Checking trained model:
    1. Whether the model suffers from High Bias (underfiting) and retrains if appropriate (increasing running time and/or increasing model complexity) to get a better score on the Training data.
    2. Once an acceptable training score is achieved, whether the model suffers from High Variance (overfitting) and retrains if appropriate (increasing regularization via dropout) to get a better score on the Validation data.
- Storing and retrieval of trained models.
- Making predictions from a trained model and logging these predictions. 
- Analysing past predictions to further understand the performance of a trained model.

## Model Evaluator

Models are evaulated using a weighted score of the Recall for the +1 and -1 classes because my objective is to make sure I don't want to miss any sharp price changes, i.e. I don't made if precision suffers and the model sometimes alerts a sharp price change when there is none - 'the boy who might cry wolf too often' because the worst that comes form this is that I buy or sell when the stock remains relatively flat. However, I want to make sure I capture as many sharp upward or downward price swings as possible as these are what can potentially cost me significantly.

## Data Chef

Raw data is retrieved using the yfinance python library to call the yahoo finance api to get daily ticker open and close prices for both the stock as well as the S&P 500.

The main steps in preparing data for input to an LSTM recurrent model are included here:
- For both the stock and for the S&P 500 index...
  - Create a timeseries of % price changes for each day the stock has traded over the past 20 years.
  - Create a class label (+1, 0 or -1) to indicate how the price closes 5 trading days in advance from that day.
  The columns in the data would now be:
  
  | Date   |      PriceChange      |  IndexPriceChange | Label  |

- Split data into Train, Validation and Test sets.
  For Timeseries models it's important not to leak future information into training data.
  Therefore Training data is taken chronologically before the Validation data which is chronologically before the Test data.

- Standardise using the distribution of the training data (which is Gaussian so a StandardScaler fits our purpose).

- Create windows of dates as input features, 
  e.g. if 20 days is chosen as the timesteps parameter then each day's % price change during the 20 previous days will become an input feature. The columns in our dataset would now be:

  | Date | PriceChange(t-20) | PriceChange(t-19) ... PriceChange(t-1) | IndexPriceChange (t-20) | IndexPriceChange (t-19) ... IndexPriceChange (t-1) | Label

- Finally, the training, validation and test data are all reshaped to be 3D array that an LSTM expects, i.e. [samples, timesteps, features], and returned in the DTO 'DataContainer'.


## Trainer

The Trainer creates the network topology according to the hyperparameters and fits the Training data, saving plots of the loss in a directory before returning the model.
Note that the data is not shuffled during training as this would destroy the ordered nature of a timeseries input. Note also that the state of a neural network is maintained whilst training a batch so the batch size as well as the number of timesteps that have been encoded as features both determine how far back data will influence the prediction. 

## Stockatron Logger

Records both predictions for a model and, at a later date, the actual result so each model might be re-evaulated in hindsight and judged whether to still be relevant.


### General Note on the Approach

Predicting stock price movements can never be perfect due to the price being dependent on the human nature of those trading with all their cognitive biases and emotional idiosyncrasies.
Given this inescapable randomness, the stochastic nature of Neural Networks models seems an appropriate choice. 
The exclusion of features such as company fundamentals is intentional as price movement and momentum seems the most important factor during the unexpected bull market that started in April 2020 during the Covid pandemic. 
The predictions use past price movements so one could say that if we assume that conditions remain similar, or follow similar patterns of randomness, then the predictions are the best guess possible.

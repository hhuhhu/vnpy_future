'''
Created on Sep 26, 2013

@author: Siviero

Script to test the predictors as to how they behave facing real market data

Algorithm:
1 - Read csv data and store it in timeframe
2 - Divide data in "base knowledge" vs. "testing"
3 - While "testing" exists, do:
  3.1 - Feed algorithm with time series
  3.2 - Compare results with testing base
  3.3 - Incorporate these test results into base knowledge
  
At the end of it, we expect to have a time series of predictions to compare 
with the testing data (benchmark) and evaluate effectiveness of prediction 
process.
'''

# Includes
from lee_predictor_time import *
from generalized_lee_predictor import *

# Imports
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
  # Reading the CSV with Pandas
  d_data = pd.read_csv('AUDUSDDaily.csv', dayfirst = True, sep = '\t', index_col = 'DATE', parse_dates = True);
  # Removing NaNs, forward-fill
  for s_key in d_data.columns:
    d_data[s_key] = d_data[s_key].fillna(method='ffill')
    d_data[s_key] = d_data[s_key].fillna(method='bfill')
    d_data[s_key] = d_data[s_key].fillna(1.0)
  column = "EMA(20)";
  # Dividing the data set
  nRows = len(d_data[column]);
  testing_pct = 0.75;
  limit = int(np.floor(nRows * testing_pct));
  print "Column " + column + " has " + str(nRows) + " rows";
  print str(testing_pct*100) + "% (" + str(limit) + ") will be used as base knowledege";
  
  base_knowledge_series = pd.DataFrame(d_data[column][:limit]);
  testing_series = pd.DataFrame(d_data[column][limit+1:]);
  
  
  # Forecasting Loop goes here
  results_columns = ['testing_base','predicted_result','error'];
  df_results = pd.DataFrame(index = testing_series.index, columns = results_columns);
  df_results['testing_base'] = testing_series;
  print df_results
  # 1 - Feed predictor function with base_knowledge
  # 2 - Store answer in a separate DataFrame
  # 3 - Insert test value into base_knowledge
  # 4 - Repeat until test is over
  #for i in df_results['testing_base']
  
  n_predictions = 1;
  order=3;
  
  for i in range(len(testing_series)):
    #print df_results['testing_base'][i]
    _pred = lee_predictor_time(base_knowledge_series[column],nPredictions=n_predictions,order=order);
    #print _pred;
    base_knowledge_series = base_knowledge_series.append({column : pd.Series(df_results['testing_base'][i])},ignore_index=True); 
    df_results['predicted_result'][i:i+n_predictions] = _pred; 
    #print base_knowledge_series
  print df_results.to_string();
  
  # Plotting
  plt.clf();
  df_results['testing_base'].plot(color = 'b');
  df_results['predicted_result'].plot(color = 'r');
  plt.show();
  
  #print d_data[column]
  #print base_knowledge_series[column]
  #print lee_predictor_time(base_knowledge_series[column],nPredictions=1);
  
  
  # This line is here for debugging purposes, will be removed later
  a=1;

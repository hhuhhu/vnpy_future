'''
Created on Oct 21, 2013

@author: Siviero
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

def fuzzy_class(val,partition_size,umin):
  return np.ceil((val-umin)/partition_size)

def leu_euclidean(x1,x2,y1,y2):
  return np.sqrt(np.dot(x1-x2,x1-x2) + np.dot(y1-y2,y1-y2))

def midpoint_fuzzy_class(val,partition_size,umin):
  return umin + (val-0.5)*partition_size;  

# TODO include secondary_series into account
def fuzzify_historical_data(data,base_series,partition_size,umin,order,nPredictions,secondary_series = 'secondary_series'):  
  input_time_series = data[base_series].copy();  
  input_time_series = pd.DataFrame(input_time_series)
  #print input_time_series
  input_time_series['Fuzzy Class'] = range(len(input_time_series[base_series]))
  
  secondary_time_series = data['secondary_series'].copy()
  secondary_time_series = pd.DataFrame(secondary_time_series);
  secondary_time_series['Fuzzy Class'] = range(len(secondary_time_series['secondary_series']));
  
  _secmin = data['secondary_series'].min()

  #secondary_time_series['Fuzzy Class'] = range(len(input_time_series[base_series]))
  for i in range(len(input_time_series[base_series])):
    input_time_series['Fuzzy Class'][i] = fuzzy_class(input_time_series[base_series][i], partition_size, umin)
    secondary_time_series['Fuzzy Class'][i] = fuzzy_class(data['secondary_series'][i], partition_size, _secmin)
  #print input_time_series.to_string()
  
  # Prediction

  # Object of interest : Relationship + RHS Fuzzy class + ED
  x1 = [];
  y1 = [];
  total = len(input_time_series);
  
  for i in range(order) :
    x1.append(input_time_series['Fuzzy Class'][total-(i*nPredictions)-1]);  
    y1.append(secondary_time_series['Fuzzy Class'][total-(i*nPredictions)-1]);  
  x1.reverse();
  y1.reverse();
  
  
  historical_fuzzified_data = [];
  for i in range(len(input_time_series[base_series])-(order+1)*nPredictions):
    _rel_lhs_classes = [];
    _rel_sec_lhs_classes = [];
    for j in range(order):
      _rel_lhs_classes.append(input_time_series['Fuzzy Class'][i+j*nPredictions])
      _rel_sec_lhs_classes.append(secondary_time_series['Fuzzy Class'][i+j*nPredictions])
    _rel_lhs_classes = np.array(_rel_lhs_classes);
    _rel_sec_lhs_classes = np.array(_rel_sec_lhs_classes);
    #_rel_lhs_classes = input_time_series['Fuzzy Class'][i:i+order].values;
    #_rel_sec_lhs_classes = secondary_time_series['Fuzzy Class'][i:i+order].values 
    _dist = leu_euclidean(x1,
                          _rel_lhs_classes,
                          y1,
                          _rel_sec_lhs_classes);
    _rel_rhs = input_time_series['Fuzzy Class'][i+order+1];    
    historical_fuzzified_data.append(dict(zip(['Relationship','ED','RHS'],(_rel_lhs_classes,_dist,_rel_rhs))));
  
  # Sort by ED
  return sorted(historical_fuzzified_data, key=lambda k: k['ED'])
  
  

def leu_predictor(data,base_series,partition_size = 0.01,scale_factor = 1,nPredictions=1,k_limit = 5,order=3):
  
  tested_series = [x for x in data.columns if x != base_series];    
  for index in tested_series:
    #print "Testing Correlation Significance for " + index;
    corr_coeff_test = scipy.stats.pearsonr(data[base_series], data[index])
    #print "p = " + str(corr_coeff_test[1]);
    if corr_coeff_test[1] >= 0.05 : # Indexes are not correlated
      del data[index]
  
  tested_series = [x for x in data.columns if x != base_series];
  
  pca_indexes = [];
  if tested_series != [] :
    # Calculate correlation matrix        
    corr_matrix = np.corrcoef(data,rowvar=0)    
    eigval,eigvec = np.linalg.eig(corr_matrix);
    #print "PCA";
    #print corr_matrix
#     print eigval;
#     print eigvec;
#     print dict(zip(tested_series,eigvec.T[:1][0][1:]))
    pca_indexes = np.array(eigvec.T[:1][0][1:],ndmin=2);
  
  """
    TODO: Iplement PCA influence on prediction
  """
  secondary_data = data.copy();
  del secondary_data[base_series];
  
  # Normalize data
  #print secondary_data.mean()
  secondary_data = (secondary_data - secondary_data.mean())/secondary_data.std()  
  data['secondary_series'] = secondary_data.dot(np.transpose(pca_indexes))
  
  #print data['secondary_series']
  # Divide main series universe of discourse in fuzzy sets
  umin = np.floor(min(d_data[base_series])/scale_factor) * scale_factor;
  
  # Build database
  sorted_historical_fuzzified_data = fuzzify_historical_data(data, base_series, partition_size, umin, order, nPredictions, secondary_series = 'secondary_series'); 
  
  """
    Prediction process as described in
    'A distance-based fuzzy time series model for exchange rates forecasting'
    by Yungho Leu, Chien-Pang Lee, Yie-Zu Jou
    in Expert Systems with Applications 36 (2009) 8107-8114
  """
    
  # Extended forecasting
  pred_results = [];  
  
  

  # Forecast one, incorporate into database, rebuild FLRs, forecast again 
  #for i in range(nPredictions):
  #print "AQUI"
  midpoint_vector = [];
  euclidean_distance_vector = [];
  # Dataset is already sorted by ED, so we fetch the k_limit first ones
  for j in range(k_limit) :
    midpoint_vector.append(midpoint_fuzzy_class(sorted_historical_fuzzified_data[j]['RHS'],partition_size, umin))
    euclidean_distance_vector.append(sorted_historical_fuzzified_data[j]['ED'])
  
  midpoint_vector = np.array(midpoint_vector);
  euclidean_distance_vector = np.array(euclidean_distance_vector);
  
  # Check if any euclidean distance is zero, if so, forecasted is the mean of 
  # such endpoints
  ii = np.where(euclidean_distance_vector == 0)[0]
  
  if ii.size == 0 :      
    w_factor = np.sum(1/euclidean_distance_vector);
    forecasted = 1/w_factor * np.sum(midpoint_vector/euclidean_distance_vector);
  else :
    _to_avg = [];
    for j in ii :
      _to_avg.append(midpoint_vector[ii[j]]);
    forecasted = np.average(_to_avg);
  
  # Incorporate predicted data into dataset
  data = data.append({base_series : forecasted},ignore_index=True) 
  umin = np.floor(min(data[base_series])/scale_factor) * scale_factor;
  #sorted_historical_fuzzified_data = fuzzify_historical_data(data, base_series, partition_size, umin, order);
  
  # Build return list
  pred_results.append(forecasted);
    
  return pred_results;


if __name__ == "__main__":
  
  # Read data
  d_data = pd.read_csv('usd_indexes.csv', dayfirst = True, index_col = 'DATE', sep='\t', parse_dates = True);
  
  # Removing NaNs, forward-fill
  for s_key in d_data.columns:
    d_data[s_key] = d_data[s_key].fillna(method='ffill')
    d_data[s_key] = d_data[s_key].fillna(method='bfill')
    d_data[s_key] = d_data[s_key].fillna(1.0)
  
  
  base_series = 'USDJPY'
  #_pred = leu_predictor(d_data,base_series,partition_size=0.1,scale_factor = 10,nPredictions=1,k_limit = 10,order=3)
  #print _pred
  
  nRows = len(d_data[base_series]);
  testing_pct = 0.75;
  limit = int(np.floor(nRows * testing_pct));
  print "Column " + base_series + " has " + str(nRows) + " rows";
  print str(testing_pct*100) + "% (" + str(limit) + ") will be used as base knowledege";
  
  base_knowledge_series = pd.DataFrame(d_data[:limit]);
  testing_series = pd.DataFrame(d_data[limit+1:]);
  
  # Forecasting Loop goes here
  results_columns = ['testing_base','predicted_result','error'];
  df_results = pd.DataFrame(index = testing_series.index, columns = results_columns);
  df_results['testing_base'] = testing_series[base_series];
  #print df_results
  # 1 - Feed predictor function with base_knowledge
  # 2 - Store answer in a separate DataFrame
  # 3 - Insert test value into base_knowledge
  # 4 - Repeat until test is over
  #for i in df_results['testing_base']
  
  n_predictions = 1;
  for i in range(len(testing_series)):    
    _pred = leu_predictor(base_knowledge_series,base_series,partition_size=0.1,scale_factor = 10,nPredictions=1,k_limit = 10,order=3)
    base_knowledge_series = base_knowledge_series.append({base_series : pd.Series(df_results['testing_base'][i])},ignore_index=True); 
    df_results['predicted_result'][i:i+n_predictions] = _pred; 
    
  df_results['error'] = df_results['predicted_result'] - df_results['testing_base'];
  df_results.to_csv('result_dbfts', '\t');
  
  # Plotting
  plt.clf();
  plt.legend(["Teste","Predicao"]);
  df_results['testing_base'].plot(color = 'b');
  df_results['predicted_result'].plot(color = 'r');
  plt.show();

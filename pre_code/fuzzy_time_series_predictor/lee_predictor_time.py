'''
Created on Aug 26, 2013

@author: Siviero
'''
import math
from operator import itemgetter, attrgetter
from numpy import *
import scipy as Sci
import scipy.linalg

from general_functions import *
from input_data import *

"""
def fetch_fuzzy_class(val,u_v):
  for u_min,u_max in u_v:
    if(val >= u_min and val <= u_max):
      dict_object = {'actual_data' : val, 'fuzzy_class' : u_v.index((u_min,u_max))};
      return dict_object;

def fetch_fuzzy_relations(val,fuzzy_relation_vector):
  r_list = []
  for i in range(len(fuzzy_relation_vector)):
    if(fuzzy_relation_vector[i][0] == val):
      print fuzzy_relation_vector[i];
      r_list.append(fuzzy_relation_vector[i]);
  return r_list;       
"""
def main():
  # 1: Define the universe of discourse
  # Method: Round min and max to thousand
  partition_size = 0.001;
  umin = math.floor(min(input_time_series));
  umax = math.ceil(max(input_time_series));
  universe = (umin,umax);
  print(universe);
  # 2: Partition of universe
  # Method: Dividing in the half-thousands
  nIter = int((umax-umin)/partition_size);
  print(nIter);
  u_vectorized = [];
  for i in range(nIter) :
    u_vectorized.append((umin + i*partition_size,umin + (i+1)*partition_size));
  print u_vectorized;  
  # 3: Analyse historical data, putting its values in the intervals
  historical_data_fuzzified = [];
  for val in input_time_series:
    historical_data_fuzzified.append(fetch_fuzzy_class(val, u_vectorized));
  print(historical_data_fuzzified);
  # 4: Establish the relations between fuzzy classes
  # In Lee predictor, number of occurrences and their chronological order are relevant,
  # which is why duplicates are not simply removed
  historical_relations_fuzzy = [];
  historical_weights = {};
  for i in range(len(historical_data_fuzzified)-1):
    _pair = (historical_data_fuzzified[i].get('fuzzy_class'),historical_data_fuzzified[i+1].get('fuzzy_class'));    
    historical_weights[_pair] = i;
    historical_relations_fuzzy.append((historical_data_fuzzified[i].get('fuzzy_class'),historical_data_fuzzified[i+1].get('fuzzy_class')));
    
  
  historical_relations_fuzzy = sorted(historical_relations_fuzzy,key = itemgetter(0,1));
  historical_relations_fuzzy_weighted = [(x,float(historical_relations_fuzzy.count(x))) for x in historical_relations_fuzzy];
  historical_relations_fuzzy_weighted = sorted(list(set(historical_relations_fuzzy_weighted)),key = itemgetter(0,1));
  
  # This could be activated, although for Alabama Enrollment time series it decreased
  # the predictor accuracy
  #print historical_relations_fuzzy_weighted;
  #historical_relations_fuzzy_weighted = [(x[0],x[1]+historical_weights[x[0]]) for x in historical_relations_fuzzy_weighted];
  #print historical_relations_fuzzy_weighted;
  #print _teste;
  
  # Implementation of Lee et al. propposed method, as described in
  # 'Modified Weighted for Enrollment Forecasting Based on Fuzzy Time Series'
  # by Muhammad Hisyam Lee, Riswan Efendi & Zuhaimy Ismail
  # in MATEMATIKA, 2009, Volume 25, Number 1, 67-78
  for j in range(len(historical_data_fuzzified[1:])):
    val = historical_data_fuzzified[j];
    
    weight_list = mat([i[1] for i in historical_relations_fuzzy_weighted if(i[0][0] == val.get('fuzzy_class'))]);
    weight_list /= sum(weight_list);
    mid_points = mat([0.5*(u_vectorized[i[0][1]][0] + u_vectorized[i[0][1]][1]) for i in historical_relations_fuzzy_weighted if(i[0][0] == val.get('fuzzy_class'))]);
    f_old = float((weight_list*mid_points.T)[0][0]);
    _tmp = u_vectorized[val.get('fuzzy_class')];
    diff = input_time_series[j] - (0.5*(_tmp[0] + _tmp[1]));
    historical_data_fuzzified[j+1]['forecasted_data'] = f_old + diff;
      
  # Graph Plotting
  plot_comparison_graph(historical_data_fuzzified,'Lee, Ismail e Efendi',['actual','forecasted']);

def generate_tuple(values):
  _list = [];
  for i in range(len(values)):
    _list.append(values[i].get('fuzzy_class'));
  return tuple(_list);
  
  
def lee_predictor_time(input_time_series,nPredictions = 1,order = 1):  
  # 1: Define the universe of discourse
  # Method: Round min and max to thousand
  partition_size = 0.001;
  umin = math.floor(min(input_time_series));
  umax = math.ceil(max(input_time_series));
  # 2: Partition of universe
  # Method: Dividing in the half-thousands
  nIter = int((umax-umin)/partition_size);
  u_vectorized = [];
  for i in range(nIter) :
    u_vectorized.append((umin + i*partition_size,umin + (i+1)*partition_size));
  # 3: Analyse historical data, putting its values in the intervals
  historical_data_fuzzified = [];
  for val in input_time_series:
    historical_data_fuzzified.append(fetch_fuzzy_class(val, u_vectorized));
  # 4: Establish the relations between fuzzy classes
  # In Lee predictor, number of occurrences and their chronological order are relevant,
  # which is why duplicates are not simply removed
  historical_relations_fuzzy = [];
  historical_weights = {};
  for i in range(len(historical_data_fuzzified)-order):
    
    _pair = [];
    for j in range(order):
      _pair.append(historical_data_fuzzified[i+j].get('fuzzy_class'));
    _pair = tuple(_pair);    
    _composed_relation = (_pair,historical_data_fuzzified[i+order].get('fuzzy_class'));
    #historical_weights[_pair] = i;
    historical_relations_fuzzy.append(_composed_relation);
    
    #_pair = (historical_data_fuzzified[i].get('fuzzy_class'),historical_data_fuzzified[i+1].get('fuzzy_class'));    
    historical_weights[_pair] = i;
    historical_relations_fuzzy.append(_composed_relation);
    
  
  historical_relations_fuzzy = sorted(historical_relations_fuzzy,key = itemgetter(0,1));
  historical_relations_fuzzy_weighted = [(x,float(historical_relations_fuzzy.count(x))) for x in historical_relations_fuzzy];
  historical_relations_fuzzy_weighted = sorted(list(set(historical_relations_fuzzy_weighted)),key = itemgetter(0,1));
  
  # This could be activated, although for Alabama Enrollment time series it decreased
  # the predictor accuracy
  #print historical_relations_fuzzy_weighted;
  #historical_relations_fuzzy_weighted = [(x[0],x[1]+historical_weights[x[0]]) for x in historical_relations_fuzzy_weighted];
  #print historical_relations_fuzzy_weighted;
  #print _teste;
  
  # Implementation of Lee et al. propposed method, as described in
  # 'Modified Weighted for Enrollment Forecasting Based on Fuzzy Time Series'
  # by Muhammad Hisyam Lee, Riswan Efendi & Zuhaimy Ismail
  # in MATEMATIKA, 2009, Volume 25, Number 1, 67-78
  """for j in range(len(historical_data_fuzzified[1:])):
    val = historical_data_fuzzified[j];
    
    weight_list = mat([i[1] for i in historical_relations_fuzzy_weighted if(i[0][0] == val.get('fuzzy_class'))]);
    weight_list /= sum(weight_list);
    mid_points = mat([0.5*(u_vectorized[i[0][1]][0] + u_vectorized[i[0][1]][1]) for i in historical_relations_fuzzy_weighted if(i[0][0] == val.get('fuzzy_class'))]);
    f_old = float((weight_list*mid_points.T)[0][0]);
    _tmp = u_vectorized[val.get('fuzzy_class')];
    diff = input_time_series[j] - (0.5*(_tmp[0] + _tmp[1]));
    historical_data_fuzzified[j+1]['forecasted_data'] = f_old + diff;
  """
  
  # Predict next one
  # Last data is all we have, so we focus on it
  pred_results = [];
  val = historical_data_fuzzified[len(historical_data_fuzzified)-order-1:len(historical_data_fuzzified)-1];
  
  # Prediction loop  
  for i in range(nPredictions):
    _new_obj = {};    
    _tuple = generate_tuple(val);
    weight_list = mat([i[1] for i in historical_relations_fuzzy_weighted if(i[0][0] == _tuple)]);
    weight_list /= sum(weight_list);
    mid_points = mat([0.5*(u_vectorized[i[0][1]][0] + u_vectorized[i[0][1]][1]) for i in historical_relations_fuzzy_weighted if(i[0][0] == _tuple)]);
    f_old = float((weight_list*mid_points.T)[0][0]);
    _tmp = u_vectorized[val[order-1].get('fuzzy_class')];
    diff = val[order-1].get('actual_data') - (0.5*(_tmp[0] + _tmp[1]));
    #_new_obj['forecasted_data'] = f_old + diff;
    #_new_obj['actual_data'] = f_old + diff;
    _new_obj['forecasted_data'] = f_old + diff;
    pred_results.append(f_old + diff);
    val = fetch_fuzzy_class(_new_obj['forecasted_data'], u_vectorized);
  return pred_results; 
  # Graph Plotting
  #plot_comparison_graph(historical_data_fuzzified,'Lee, Ismail e Efendi',['actual','forecasted']);

  
if __name__ == '__main__':
    #main();
    print lee_predictor_time(audusd_ema20_float64,nPredictions=1,order=3);
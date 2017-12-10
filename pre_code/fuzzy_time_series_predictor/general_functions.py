'''
Created on Aug 27, 2013

@author: Siviero
'''
import matplotlib.pyplot as plt
import math

def fetch_fuzzy_class(val,u_v):
  for u_min,u_max in u_v:
    if(val >= u_min and val <= u_max):
      dict_object = {'actual_data' : val, 'fuzzy_class' : u_v.index((u_min,u_max))};
      return dict_object;

def fetch_fuzzy_relations(val,fuzzy_relation_vector):
  r_list = []
  for i in range(len(fuzzy_relation_vector)):
    if(fuzzy_relation_vector[i][0] == val):
      print(fuzzy_relation_vector[i])
      r_list.append(fuzzy_relation_vector[i]);
  return r_list;       

def get_midpoint(ptuple):
  return 0.5*(ptuple[0]+ptuple[1]);

def get_midpoint_vector(tuple_vector):
  return [get_midpoint(x) for x in tuple_vector];

def get_length(ptuple):
  return ptuple[1]-ptuple[0];
'''def approximate_universe_exp(l_values):
  
  umin = min(l_values);
  _min_pot = 10**(math.ceil(math.log10(umin))); 
  umax = max(l_values);
  _max_post 
  universe = (umin,umax);
  '''

def plot_comparison_graph(historical_data_forecasted,title,legend):
  actual = [x.get('actual_data') for x in historical_data_forecasted[1:]];
  predicted = [x.get('forecasted_data') for x in historical_data_forecasted[1:]];
  plt.plot(range(len(actual)),actual,'r',range(len(actual)),predicted,'b');
  plt.title(title);
  plt.legend(legend);
  plt.show();
  
def fuzzy_pertinence_in_tuple(value,ptuple,tuple_list):
  if(tuple_list.index(ptuple) == 0 and value <= get_midpoint(ptuple)) :
    return 1.0;
  if(tuple_list.index(ptuple) == len(tuple_list)-1 and value >= get_midpoint(ptuple)) :
    return 1.0;  
  return max(0,1-abs(value - get_midpoint(ptuple))/(2*get_length(ptuple)));

def generalize_discard_irrelevant_values(matrix,max_n):
  
  for i in range(len(matrix)):
    #print matrix[i];
    sor = (sorted(matrix[i],reverse = True));
    matrix[i] = [x if x >= sor[max_n-1] else 0 for x in matrix[i]];
  return matrix;  
      
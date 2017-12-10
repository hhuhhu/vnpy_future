'''
Created on Sep 5, 2013

@author: Siviero
'''
'''
Created on Aug 27, 2013

@author: Siviero
'''
import math
from operator import itemgetter, attrgetter
from numpy import *
import scipy as Sci
import scipy.linalg
from sklearn.preprocessing import normalize

from pre_code.fuzzy_time_series_predictor.general_functions import *
from pre_code.fuzzy_time_series_predictor.input_data import *


def high_order_predicition(fuzzy_logical_relationship, midpoint_vector):
    # print fuzzy_logical_relationship
    triple_tuple = fuzzy_logical_relationship[0]
    target = fuzzy_logical_relationship[1]
    # print triple_tuple
    # print target
    if (target == 0):
        val = (midpoint_vector[0] + 0.5 * midpoint_vector[1]) / 1.5
    elif (target < len(midpoint_vector) - 1):
        val = (0.5 * midpoint_vector[target - 1] + midpoint_vector[target] + 0.5 * midpoint_vector[target + 1]) / 2
    else:
        val = (0.5 * midpoint_vector[target - 1] + midpoint_vector[target]) / 1.5
    return val


def main():
    first_factor_time_series = taipei_temp_time_series
    second_factor_time_series = taipei_cloud_dens_time_series

    # 1: Define the universe of discourse of first and second factors
    # Method: Round min and max to unit
    # First
    u_partition_size = 1
    umin = math.floor(min(first_factor_time_series))
    umax = math.ceil(max(first_factor_time_series))
    u_universe = (umin, umax)
    # print(u_universe)
    # Second
    v_partition_size = 10
    vmin = math.floor(min(second_factor_time_series) / v_partition_size) * v_partition_size
    vmax = math.ceil(max(second_factor_time_series) / v_partition_size) * v_partition_size
    v_universe = (vmin, vmax)
    # print(v_universe)

    # 2: Partition of universe
    # Method: Dividing in partition sizes
    # First
    nIter = int((umax - umin) / u_partition_size)
    # print(nIter)
    u_vectorized = []
    for i in range(nIter):
        u_vectorized.append((umin + i * u_partition_size, umin + (i + 1) * u_partition_size))
    # print u_vectorized
    # First
    nIter = int((vmax - vmin) / v_partition_size)
    # print(nIter)
    v_vectorized = []
    for i in range(nIter):
        v_vectorized.append((vmin + i * v_partition_size, vmin + (i + 1) * v_partition_size))
    # print v_vectorized

    # 3: Analyse historical data, putting its values in the intervals
    first_historical_data_fuzzified = []
    for val in first_factor_time_series:
        first_historical_data_fuzzified.append(fetch_fuzzy_class(val, u_vectorized))
    # print(first_historical_data_fuzzified)

    second_historical_data_fuzzified = []
    for val in second_factor_time_series:
        second_historical_data_fuzzified.append(fetch_fuzzy_class(val, v_vectorized))
    # print(second_historical_data_fuzzified)


    # 4: Establish the relations between fuzzy classes
    # In Lee predictor, number of occurrences and their chronological order are relevant,
    # which is why duplicates are not simply removed
    historical_relations_fuzzy = []
    k = 3
    for i in range(len(first_historical_data_fuzzified) - k):

        # print(first_historical_data_fuzzified[i+k])
        _pair = []
        for j in range(k):
            _pair.append((first_historical_data_fuzzified[i + j].get('fuzzy_class'),
                          second_historical_data_fuzzified[i + j].get('fuzzy_class')))
        _composed_relation = (_pair, first_historical_data_fuzzified[i + k].get('fuzzy_class'))
        # historical_weights[_pair] = i
        historical_relations_fuzzy.append(_composed_relation)

    # 5: Prediction - Needs to be adapted to include non-existent relations
    # this is very important to prediction
    midpoint_vector = get_midpoint_vector(u_vectorized)
    for i in range(len(historical_relations_fuzzy)):
        # print historical_relations_fuzzy[i]
        # Check all relationships starting with current state
        _aux_list = [x for x in historical_relations_fuzzy if (x[0] == historical_relations_fuzzy[i][0])];
        # print _aux_list
        # Create a list of dictionaries counting the occurrences
        _dict_list = [{'fuzzy_relationship': x, 'nOccurrences': _aux_list.count(x)} for x in _aux_list];
        # print _dict_list
        # Calculate total occurrences
        total = 0
        for j in _dict_list:
            total += j.get('nOccurrences')
        # print total
        # Calculate each tj
        for j in _dict_list:
            _tj = high_order_predicition(j.get('fuzzy_relationship'), midpoint_vector)
            j['tj'] = _tj
        # print _dict_list
        # Final prediction
        soma = 0
        for j in _dict_list:
            soma += j.get('tj') * j.get('nOccurrences')
        first_historical_data_fuzzified[i + k]['forecasted_data'] = soma / total

    # Prediction to next one
    _pair = []
    i = len(first_historical_data_fuzzified) - 3
    for j in range(k):
        _pair.append((first_historical_data_fuzzified[i + j].get('fuzzy_class'),
                      second_historical_data_fuzzified[i + j].get('fuzzy_class')))
        # print _pair
    i = len(first_historical_data_fuzzified)
    _tmp = (3 * first_historical_data_fuzzified[i - 1]['forecasted_data'] + 2 * first_historical_data_fuzzified[i - 2][
        'forecasted_data'] + first_historical_data_fuzzified[i - 3]['forecasted_data']) / 6
    first_historical_data_fuzzified.append({'forecasted_data': _tmp})
    # Graph Plotting
    plot_comparison_graph(first_historical_data_fuzzified, 'Lee, Wang, Chen e Leu', ['actual', 'forecasted']);


if __name__ == '__main__':
    main()

#!/usr/bin/python3
import os, sys, getopt, re, random
import subprocess as sp
import logging as lg
import json
from datetime import datetime as dt

BASE_DIR = os.path.dirname(__file__) 
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')

DEFAULT_WEKA_PATH = os.path.join( *[BASE_DIR, 'weka', 'weka.jar'] )

HOEFFDINGTREE_CLASS = "weka.classifiers.trees.HoeffdingTree"
J48_CLASS = "weka.classifiers.trees.J48"

BAGGING_CLASS = "weka.classifiers.meta.Bagging"

BAGGING_ENABLED = True

DEFAULT_CLASS = HOEFFDINGTREE_CLASS
DEFAULT_MESURE = 'ROC Area'
DONE_PARAMS = []



def get_main_command():
        ## setting up main command
        command = "java -cp {}".format(DEFAULT_WEKA_PATH)
        if BAGGING_ENABLED:
            command = "{} {}".format(command, BAGGING_CLASS)
        else:
            command = "{} {}".format(command, DEFAULT_CLASS)
        return command

def get_weka_learning_result(raw_data):
    try:
        all_data = raw_data.decode('utf-8')
        data = all_data[ all_data.index('Stratified cross-validation') : ]

        columns_str = [line for line in data.splitlines() if line.find(DEFAULT_MESURE) != -1][0].strip()
        avgs_str = [line for line in data.splitlines() if line.find('Weighted Avg') != -1][0].strip()

        columns = re.findall('(\w+[ -]\w+|\w+)', columns_str)[:-1]
        avgs_raw = re.findall('\d+\.\d+|\?', avgs_str)
        avgs = [x if x != '?' else -1 for x in avgs_raw]
        if columns.index(DEFAULT_MESURE) != -1:
            return float(avgs[columns.index(DEFAULT_MESURE)]), data, columns, avgs_raw

    except:
        pass

    return 0, all_data, [], []


def save_data(data, mesure):
    file_path = os.path.join( OUTPUT_FOLDER, DEFAULT_CLASS + '.json')

    try:
        with open(file_path, 'r') as file:
            runs =  json.loads(file.read())

        with open(file_path, 'w') as file:
            if BAGGING_ENABLED:
                index = runs['bagging']['index']
                runs['bagging']['results'] +=  [{
                    'i': index,
                    'mesure': mesure,
                    'content': data,
                }]
                runs['bagging']['index'] += 1
            else:
                index = runs['no-bagging']['index']
                runs['no-bagging']['results'] = [{
                    'i': index,
                    'mesure': mesure,
                    'content': data,
                }]
                runs['no-bagging']['index'] += 1
            file.write(json.dumps(runs))

    except Exception as e:
            print(e)



def get_j48_params():
    global DONE_PARAMS
    params = ''
    
    while params in DONE_PARAMS or params == '':
        if bool(random.getrandbits(1)): 
            params = '{} -C {}'.format(params, random.uniform(0,1)) ## above .5 is same as disabling it
        else:
            if bool(random.getrandbits(1)): 
                params = '{} -N {} -R'.format(params, random.randint(2, 130)) # replace 130 with number of example maybe ? ask teacher for limit
            else: 
                params = '{} -U'.format(params)

        if bool(random.getrandbits(1)): 
            params = '{} -M {}'.format(params, random.randint(1, 130)) # replace 130 with number of example maybe ? ask teacher for limit

        if bool(random.getrandbits(1)): 
            params = '{} -Q {}'.format(params, random.randint(1,130)) # replace 130 with number of example maybe ? ask teacher for limit

        if bool(random.getrandbits(1)): 
            params = '{} -O'.format(params)

        if bool(random.getrandbits(1)): 
            params = '{} -B'.format(params)

        if params.find('-U') == -1:
            if bool(random.getrandbits(1)): 
                params = '{} -S'.format(params)
            if bool(random.getrandbits(1)): 
                params = '{} -L'.format(params)
            if bool(random.getrandbits(1)): 
                params = '{} -A'.format(params)
            if bool(random.getrandbits(1)): 
                params = '{} -J'.format(params)
        
        if bool(random.getrandbits(1)): 
            params = '{} -doNotMakeSplitPointActualValue'.format(params)

        if params == '':
            params = '-C 0.25 -M 7'

    DONE_PARAMS += [params]

    return params

def get_bagging_params():
    global DONE_PARAMS
    params = ''

    while params not in DONE_PARAMS and params == '':
        if bool(random.getrandbits(1)): 
            params = "{} -P {}".format(params, random.randint(0, 100))
        if bool(random.getrandbits(1)): 
            params = "{} -O".format(params)
        if bool(random.getrandbits(1)): 
            params = "{} -S {}".format(params, random.randint(1, 130))
        if bool(random.getrandbits(1)): 
            params = "{} -num-slots {}".format(params, random.randint(1, 30))
        if bool(random.getrandbits(1)): 
            params = "{} -I {}".format(params, random.randint(1, 30))
        if bool(random.getrandbits(1)): 
            params = "{} -store-out-of-bag-predictions".format(params)
        if bool(random.getrandbits(1)): 
            params = "{} -output-out-of-bag-complexity-statistics".format(params)
        if bool(random.getrandbits(1)): 
            params = "{} -represent-copies-using-weights".format(params)
                    
        if params == '':
            params = '-S 50'

    DONE_PARAMS += [params]
    return params


def run(algo_params, max_mesure, best_algo_params, best_bagging_params, best_mesures, i = 0):
    if BAGGING_ENABLED:
        bagging_params = get_bagging_params()
        final_command = "{} {} -t '{}' -W {} -- {}".format(COMMAND, bagging_params, DATA_PATH, DEFAULT_CLASS, algo_params)
    else:
        bagging_params = None
        final_command = "{} {} -t '{}'".format(COMMAND, algo_params, DATA_PATH)

    returned_raw = sp.check_output(['bash', '-c', final_command], stderr = sp.STDOUT)
    mesure, returned, mesures_cols, mesures_vals = get_weka_learning_result(returned_raw)

    if mesure > max_mesure:
        max_mesure = mesure
        best_algo_params = algo_params
        if BAGGING_ENABLED:
            best_bagging_params = bagging_params
        best_mesures = [mesures_cols, mesures_vals]

    os.system('clear')
    print('Run {} => {} {}'.format(str(i).zfill(4), DEFAULT_MESURE, max_mesure), end = '\n\n')
    if BAGGING_ENABLED:
        print('Bagging Parameters : {}'.format(best_bagging_params))
    print('Algorithme Parameters : {}'.format(best_algo_params), end = '\n\n')
    print('Mesures :')
    for col, val in list(zip(best_mesures[0], best_mesures[1])):
        print('{} : {}'.format(col, val))

    save_data(returned, mesure)

    return max_mesure, best_algo_params, best_bagging_params, best_mesures


def learn():
    max_mesure = 0
    best_algo_params = ''
    best_bagging_params = ''
    best_mesures = [[],[]]

    i = 0
    while True:
        i += 1
        algo_params = get_hoeffdingtree_params()

        try:
           max_mesure, best_algo_params, best_bagging_params, best_mesures = run(algo_params, max_mesure, best_algo_params, best_bagging_params, best_mesures, i)

        except Exception as e:
            lg.exception(e)

def get_hoeffdingtree_params():
    global DONE_PARAMS
    params = ''
    
    while params in DONE_PARAMS or params == '':
        if bool(random.getrandbits(1)): 
            params = '{} -H {}'.format(params, random.uniform(0,1))
        if bool(random.getrandbits(1)): 
            params = '{} -L {}'.format(params, random.choice([0,1,2]))
        if bool(random.getrandbits(1)): 
            params = '{} -S {}'.format(params, random.choice([0,1]))
        if bool(random.getrandbits(1)): 
            params = '{} -M {}'.format(params, random.uniform(0,1))
        if bool(random.getrandbits(1)): 
            params = '{} -G {}'.format(params, random.randint(0,300))
        if bool(random.getrandbits(1)): 
            params = '{} -N {}'.format(params, random.randint(0,130))
        if bool(random.getrandbits(1)): 
            params = '{} -P'.format(params)
        
        if params == '':
            params = '-H 0.05 -L 2 -S 1 -M 0.01 -G 200 -N 0'
            
    DONE_PARAMS += [params]
    return params
# 
#  -S 1 -E 1.0 e-7 -H 0.05 -L 2 -S 1 -M 0.01 G 200 N 0.0
# java -cp weka/weka.jar/ weka.classifiers.trees.HoeffdingTree -H 0.05 -L 2 -S 1 -M 0.01 -G 200 -N 0 -t 'datasets/features/train/y_features_opt2_300_train____________.csv'
if __name__ == '__main__':
    DATA_PATH = os.path.join(*[BASE_DIR, 'datasets', 'features', 'train', '2classes', 'y_features_opt2_300_train.csv'])
    COMMAND = get_main_command()
    learn()
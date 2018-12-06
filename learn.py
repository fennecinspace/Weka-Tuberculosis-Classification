#!/usr/bin/python3
import os, sys, getopt, re, random
import subprocess as sp
import logging as lg
import json
from datetime import datetime as dt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_DATA_PATH = os.path.join(*[BASE_DIR, 'datasets', 'features', 'train', '2classes', 'y_features_opt2_300_train.csv'])
DEFAULT_TEST_DATA_PATH = os.path.join(*[BASE_DIR, 'datasets', 'features', 'val', '2classes', 'y_features_opt2_300_val.csv'])
DEFAULT_OUTPUT_PATH = os.path.join(*[BASE_DIR, 'output'])

DEFAULT_WEKA_PATH = os.path.join( *[BASE_DIR, 'weka', 'weka.jar'] )

HOEFFDINGTREE_CLASS = "weka.classifiers.trees.HoeffdingTree"
J48_CLASS = "weka.classifiers.trees.J48"
RANDOMFOREST_CLASS = "weka.classifiers.trees.RandomForest"
BAGGING_CLASS = "weka.classifiers.meta.Bagging"

DEFAULT_ALGO_CLASS = J48_CLASS
DEFAULT_MESURE = 'ROC Area'
DONE_PARAMS = []

def get_args(argv):
    data, test, output, weka_path, weka_class = None, None, None, None, None
    bypass_settings, bagging_enabled = False, False

    try:
        if '-c' in argv:
            weka_class = argv[argv.index('-c') + 1]
        elif '--weka-class' in argv:
            weka_class = argv[argv.index('--weka-class') + 1]
    except: print('"-c" has been passed incorrectly do "-h" for HELP')
    
    try:
        if '-p' in argv:
            weka_path = argv[argv.index('-p') + 1]
        elif '--weka_path' in argv:
            weka_path = argv[argv.index('--weka_path') + 1]
    except: print('"-p" has been passed incorrectly do "-h" for HELP')
    
    try:
        if '-o' in argv:
            output = argv[argv.index('-o') + 1]
        elif '--output' in argv:
            output = argv[argv.index('--output') + 1]
    except: print('"-o" has been passed incorrectly do "-h" for HELP')
    

    try:
        if '-d' in argv:
            data = argv[argv.index('-d') + 1]
        elif '--data' in argv:
            data = argv[argv.index('--data') + 1]
    except: print('"-d" has been passed incorrectly do "-h" for HELP')

    try:
        if '-t' in argv:
            test = argv[argv.index('-t') + 1]
        elif '--test' in argv:
            test = argv[argv.index('--test') + 1]
    except: print('"-t" has been passed incorrectly do "-h" for HELP')

    try:
        if '-b' in argv or '--bagging' in argv:
            bagging_enabled = True
    except: pass

    try:
        if '-y' in argv or '--yes' in argv:
            bypass_settings = True
    except: pass

    try:
        if '-h' in argv or '--help' in argv:
            print ('Returns the best model for a certain dataset using Weka.\n\npython3 classifier.py [OPTIONS]\n')
            print ('  -d, --data [path]           Training Dataset, file format must be compatibale with weka')
            print ('  -t, --test [path]           Testing Dataset, file format must be compatibale with weka')
            print ('  -o, --output [path]         Model output directory')
            print ('  -p, --weka-path [path]      Path to the weka.jar file')
            print ('  -c, --weka-class [class]    Weka class, choose between:')
            print ('                                  - weka.classifiers.trees.J48 (default)')
            print ('                                  - weka.classifiers.trees.HoeffdingTree')
            print ('                                  - weka.classifiers.trees.RandomForest')
            print ('  -b, --bagging               Enable Bagging')
            print ('  -y, --yes                   Bypass Confirmation')
            print ('  -h, --help                  Show help\n')

    except: pass

    return data, test, output, weka_path, weka_class, bypass_settings, bagging_enabled


class Weka():
    __slots__ = ['weka_path', 'weka_class', 'bagging_enabled'] 
    
    def __init__(self, weka_path = None, weka_class = None, bagging_enabled = False):
        self.weka_path = self.check_weka_path(weka_path or DEFAULT_WEKA_PATH)
        self.weka_class = self.check_algorithm(weka_class or DEFAULT_ALGO_CLASS)
        self.bagging_enabled = bagging_enabled

    @classmethod
    def check_weka_path(cls, path):
        while not os.path.exists(path) :
            path = input("Weka could not be found, weka.jar path : ")
        return path

    @classmethod
    def check_algorithm(cls, class_name):
        while class_name not in [J48_CLASS, HOEFFDINGTREE_CLASS, RANDOMFOREST_CLASS] :
            class_name = input("Incorrect class, Please choose one of the following:\n{}\n{}\n{}\n".format(J48_CLASS, HOEFFDINGTREE_CLASS, RANDOMFOREST_CLASS))
        return class_name

class Model():
    __slots__ = ['data_path', 'test_data_path', 'output_path', 'weka', 'max_mesure', 'best_algo_params', 'best_bagging_params', 'best_mesures', 'old_runs'] 

    def __init__(self, weka, data_path, test_data_path, output_path = None):
        self.max_mesure = 0
        self.best_algo_params = ''
        self.best_bagging_params = ''
        self.best_mesures = [[],[]]

        self.weka = weka
        self.data_path = self.check_data_path(data_path or DEFAULT_DATA_PATH)
        self.test_data_path = self.check_test_path(test_data_path)

        file_path, file_name = os.path.split(self.data_path)
        self.output_path, self.old_runs = self.check_output_path(output_path or DEFAULT_OUTPUT_PATH, weka, file_name)


    @classmethod
    def check_output_path(cls, path, weka, file_name):
        global DONE_PARAMS
        old_runs = None

        if path:
            while True:
                if os.path.isdir(path):
                    if not os.path.exists( os.path.join(path, file_name + '_' + weka.weka_class + '.json') ): 
                        try:
                            with open(os.path.join(*[BASE_DIR, 'output', 'template.json'])) as file:
                                template = file.read()
                            with open(os.path.join(path, file_name + '_' + weka.weka_class + '.json'), 'w') as file:
                                file.write(template)
                            
                            path = os.path.join(path, file_name + '_' + weka.weka_class + '.json')
                            break
                        except Exception as e:
                            pass
                    else:
                        try:
                            with open(os.path.join(path, file_name + '_' + weka.weka_class + '.json'), 'r') as file:
                                runs = json.loads(file.read())
                            
                            if weka.bagging_enabled:
                                DONE_PARAMS = runs['bagging']['algo_done']
                                old_runs = runs['bagging']
                            else:
                                DONE_PARAMS = runs['no-bagging']['algo_done']
                                old_runs = runs['no-bagging']

                            path = os.path.join(path, file_name + '_' + weka.weka_class + '.json')
                            break
                        except Exception as e:
                            pass

                path = input("output file directory doesn't exist, output path : ")
        return path, old_runs


    @classmethod
    def check_data_path(cls, path):
        while not os.path.exists(path) :
            path = input("incorrect dataset path, dataset path : ")
        return path


    @classmethod
    def check_test_path(cls, path):
        if path:
            while not os.path.exists(path) :
                path = input("incorrect dataset path, dataset path : ")
        return path


    def save_data(self,data, mesure, algo_params, bagging_params):
        file_path = self.output_path

        try:
            with open(self.output_path, 'r') as file:
                runs =  json.loads(file.read())

            with open(self.output_path, 'w') as file:
                if self.weka.bagging_enabled:
                    index = runs['bagging']['index']
                    runs['bagging']['results'] +=  [{
                        'i': index,
                        'mesure': mesure,
                        'algo_params': algo_params,
                        'bagging_params': bagging_params,
                        'content': data,
                    }]
                    runs['bagging']['algo_done'] = DONE_PARAMS
                    runs['bagging']['index'] += 1
                else:
                    index = runs['no-bagging']['index']
                    runs['no-bagging']['results'] = [{
                        'i': index,
                        'mesure': mesure,
                        'algo_params': algo_params,
                        'content': data,
                    }]
                    runs['no-bagging']['algo_done'] = DONE_PARAMS
                    runs['no-bagging']['index'] += 1
                file.write(json.dumps(runs))

        except Exception as e:
            print(e)
            

    def get_weka_learning_result(self, raw_data, col = 'ROC Area'):
        try:
            all_data = raw_data.decode('utf-8')
            data = all_data[ all_data.index('Stratified cross-validation') : ]

            columns_str = [line for line in data.splitlines() if line.find(col) != -1][0].strip()
            avgs_str = [line for line in data.splitlines() if line.find('Weighted Avg') != -1][0].strip()

            columns = re.findall('(\w+[ -]\w+|\w+)', columns_str)[:-1]
            avgs_raw = re.findall('\d+\.\d+|\?', avgs_str)
            avgs = [x if x != '?' else -1 for x in avgs_raw]
            if columns.index(col) != -1:
                return float(avgs[columns.index(col)]), data, columns, avgs_raw
            else:
                print('Column ({}) could not be found !'.format(col))
                return 0, data, [], []
        except:
            return 0, all_data, [], []


    @classmethod
    def get_main_command(cls, weka):
        ## setting up main command
        command = "java -cp {}".format(weka.weka_path)
        if weka.bagging_enabled:
            command = "{} {}".format(command, BAGGING_CLASS)
        else:
            command = "{} {}".format(command, weka.weka_class)
        return command


    def learn(self):
        command = self.get_main_command(self.weka)
        train_path = "-t '{}'".format(self.data_path)

        i = len(DONE_PARAMS) + 1
        while True:
            if self.weka.weka_class == J48_CLASS:
                algo_params = self.get_j48_params()
            elif self.weka.weka_class == HOEFFDINGTREE_CLASS:
                algo_params = self.get_hoeffdingtree_params()
            elif self.weka.weka_class == RANDOMFOREST_CLASS:
                algo_params = self.get_RandomForest_params()
            
            self.run(command, train_path, algo_params, i)

            i += 1


    def test(self):
        command = self.get_main_command(self.weka)
        test_path = "-t '{}' -T '{}'".format(self.data_path, self.test_data_path)

        best_res = {'i': 0, 'mesure': 0}
        
        training_res = self.old_runs['results']
        for res in training_res:
            if best_res['mesure'] < res['mesure']:
                best_res['mesure'] = res['mesure']        
                best_res['algo_params'] = res['algo_params']
                if 'bagging_params' in best_res:
                    best_res['bagging_params'] = res['bagging_params']
                else:
                    best_res['bagging_params'] = None
                best_res['content'] = res['content']
        
        if best_res['mesure'] > 0:
            self.run(command, test_path, best_res['algo_params'], 0, best_res['bagging_params'])

        input('PRESS ANY, TO EXIT...')


           
    def run(self, command, data_path, algo_params, i = 0, bagging_params = None):
        if self.weka.bagging_enabled:
            if not bagging_params:
                bagging_params = self.get_bagging_params()
            final_command = "{} {} {} -W {} -- {}".format(command, bagging_params, data_path, self.weka.weka_class, algo_params)
        else:
            bagging_params = None
            final_command = "{} {} {}".format(command, algo_params, data_path)

        try:
            returned_raw = sp.check_output(['bash', '-c', final_command], stderr = sp.STDOUT)
            mesure, returned, mesures_cols, mesures_vals = self.get_weka_learning_result(returned_raw, DEFAULT_MESURE)

            if not self.test_data_path:
                self.update_best_params(mesure, mesures_cols, mesures_vals, algo_params, bagging_params)

            if self.test_data_path:
                print("\nCOMMAND USED =>  {}\n".format(final_command))
                print(returned)
            else:
                self.print_learning_progress(i, mesure)
                self.save_data(returned, mesure, algo_params, bagging_params)

        except Exception as e:
            lg.exception(e)

    
    
    def update_best_params(self, mesure, mesures_cols, mesures_vals, algo_params, bagging_params):
        if mesure > self.max_mesure:
            self.max_mesure = mesure
            self.best_algo_params = algo_params
            if self.weka.bagging_enabled:
                self.best_bagging_params = bagging_params
            self.best_mesures = [mesures_cols, mesures_vals]


    def print_learning_progress(self, run_number = 1, run_mesure = 0):
        os.system('clear')
        print('Last Run {} => {} {}'.format(str(run_number).zfill(4), DEFAULT_MESURE, run_mesure), end = '\n\n')
        print(' == Best Run == \n')
        if self.weka.bagging_enabled:
            print('Bagging Parameters : {}'.format(self.best_bagging_params))
        print('Algorithme Parameters : {}'.format(self.best_algo_params), end = '\n\n')
        print('Mesures :')
        for col, val in list(zip(self.best_mesures[0], self.best_mesures[1])):
            print('{} : {}'.format(col, val))


    def get_j48_params(self):
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

    def get_RandomForest_params(self):
        global DONE_PARAMS
        params = ''
        
        while params in DONE_PARAMS or params == '':
            if bool(random.getrandbits(1)): 
                params = '{} -attribute-importance'.format(params)
            
            if bool(random.getrandbits(1)): 
                params = '{} -P {}'.format(params, random.randint(0,100)) 
            
            if bool(random.getrandbits(1)): 
                params = '{} -I {}'.format(params, random.randint(0,500))     
            
            if bool(random.getrandbits(1)): 
                params = '{} -num-slots 0'.format(params) 
            
            if bool(random.getrandbits(1)):
                params = '{} -K {}'.format(params, random.randint(0,300))
            
            if bool(random.getrandbits(1)): 
                params = '{} -M {}'.format(params, random.randint(1,50)) 
            
            if bool(random.getrandbits(1)): 
                params = '{} -V {}'.format(params, round(random.uniform(0,1), 5))
            
            if bool(random.getrandbits(1)): 
                params = '{} -S {}'.format(params, random.randint(1,50)) 
            
            if bool(random.getrandbits(1)): 
                params = '{} -depth {}'.format(params, random.randint(0,300))
            
            if bool(random.getrandbits(1)): 
                params = '{} -N {}'.format(params, random.randint(0,50))
        
            if bool(random.getrandbits(1)): 
                params = '{} -U'.format(params)

            if bool(random.getrandbits(1)): 
                params = '{} -B'.format(params)

            if bool(random.getrandbits(1)): 
                params = '{} -O'.format(params)
            
            if bool(random.getrandbits(1)): 
                params = '{} -num-decimal-places'.format(params)    

            if bool(random.getrandbits(1)): 
                params = '{} -do-not-check-capabilities'.format(params)

            if params == '':
                params = '-P 0.25 '

        DONE_PARAMS += [params]

        return params



    def get_hoeffdingtree_params(self):
        global DONE_PARAMS
        params = ''
        
        while params in DONE_PARAMS or params == '':
            if bool(random.getrandbits(1)): 
                params = '{} -H {}'.format(params, round(random.uniform(0,1), 5))
            if bool(random.getrandbits(1)): 
                params = '{} -L {}'.format(params, random.choice([0,1,2]))
            if bool(random.getrandbits(1)): 
                params = '{} -E {}'.format(params, round(random.uniform(0,1), 5))
            if bool(random.getrandbits(1)): 
                params = '{} -S {}'.format(params, random.choice([0,1]))
            if bool(random.getrandbits(1)): 
                params = '{} -M {}'.format(params, round(random.uniform(0,1), 5))
            if bool(random.getrandbits(1)): 
                params = '{} -G {}'.format(params, random.randint(0,300))
            if bool(random.getrandbits(1)): 
                params = '{} -N {}'.format(params, random.randint(0,130))
            if bool(random.getrandbits(1)): 
                params = '{} -P'.format(params)

            if params == '':
                params = '-H 0.05 -L 2 -E 0.00091 -S 1 -M 0.01 -G 200 -N 0'
                
        DONE_PARAMS += [params]
        return params


    def get_bagging_params(self):
        params = ''

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

        return params

#!/usr/bin/python3
import os, sys, getopt, re, random
import subprocess as sp
import logging as lg
import json
from datetime import datetime as dt

BASE_DIR = os.path.dirname(__file__) 

DEFAULT_DATA_PATH = os.path.join(*[BASE_DIR, 'datasets', 'features', 'train', '2classes', 'y_features_opt2_300_train.csv'])
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
    data, output, weka_path, weka_class = None, None, None, None
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
            print ('  -d, --data [path]           Dataset to use, file format must be compatibale with weka')
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
    

    # try:
    #     opts, args = getopt.getopt(argv,"d:o:p:c",["data=","output=","weka-path=","weka-class="])
    # except getopt.GetoptError:
    #     print ('python3 learn.py [OPTIONS]\nReturns the best model for a certain dataset, uses Weka\n')
    #     print ('  -d, --data           dataset to use, file format must be compatibale with weka')
    #     print ('  -o, --output         model output file (must be .json, parent directory must exist)')
    #     print ('  -p, --weka-path      path to the weka.jar file')
    #     print ('  -c, --weka-class     weka class')
    #     sys.exit(2)
    # for opt, arg in opts:
    #     print(opt, arg)
    #     if opt in ["-d", "--data"]:
    #         data = arg
    #     elif opt in ["-o", "--output"]:
    #         output = arg
    #     elif opt in ["-p", "--weka-path"]:
    #         weka_path = arg
    #     elif opt in ["-c", "--weka-class"]:
    #         weka_class = arg

    return data, output, weka_path, weka_class, bypass_settings, bagging_enabled


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
    __slots__ = ['data_path', 'output_path', 'weka'] 

    def __init__(self, weka, data_path, output_path = None):
        self.weka = weka
        self.data_path = self.check_data_path(data_path or DEFAULT_DATA_PATH)
        self.output_path = self.check_output_path(output_path or DEFAULT_OUTPUT_PATH, weka)

    @classmethod
    def check_output_path(cls, path, weka):
        if path:
            while True:
                if os.path.isdir(path):
                    if not os.path.exists( os.path.join(path, weka.weka_class + '.json') ): 
                        try:
                            with open(os.path.join(path, weka.weka_class + '.json'), 'w') as file:
                                file.write('{ "bagging": { "index": 0, "results": [] }, "no-bagging": { "index": 0, "results": [] } }')
                            break
                        except Exception as e:
                            pass
                    else:
                        break
                path = input("output file directory doesn't exist, output path : ")
        return path

    @classmethod
    def check_data_path(cls, path, val = False):
        while not os.path.exists(path) :
            path = input("incorrect dataset path, dataset path : ")
        return path

    def save_data(self,data, mesure, algo_params, bagging_params):
        file_path = os.path.join( self.output_path, self.weka.weka_class + '.json')

        try:
            with open(file_path, 'r') as file:
                runs =  json.loads(file.read())

            with open(file_path, 'w') as file:
                if self.weka.bagging_enabled:
                    index = runs['bagging']['index']
                    runs['bagging']['results'] +=  [{
                        'i': index,
                        'mesure': mesure,
                        'algo_params': algo_params,
                        'bagging_params': bagging_params,
                        'content': data,
                    }]
                    runs['bagging']['index'] += 1
                else:
                    index = runs['no-bagging']['index']
                    runs['no-bagging']['results'] = [{
                        'i': index,
                        'mesure': mesure,
                        'algo_params': algo_params,
                        'content': data,
                    }]
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
        max_mesure = 0
        best_algo_params = ''
        best_bagging_params = ''
        best_mesures = [[],[]]

        command = self.get_main_command(self.weka)

        i = 0
        while True:
            i += 1
            if self.weka.weka_class == J48_CLASS:
                algo_params = self.get_j48_params()
            elif self.weka.weka_class == HOEFFDINGTREE_CLASS:
                algo_params = self.get_hoeffdingtree_params()
            elif self.weka.weka_class == RANDOMFOREST_CLASS:
                algo_params = self.get_RandomForest_params()

            if self.weka.bagging_enabled:
                bagging_params = self.get_bagging_params()
                final_command = "{} {} -t '{}' -W {} -- {}".format(command, bagging_params, self.data_path, self.weka.weka_class, algo_params)
            else:
                bagging_params = None
                final_command = "{} {} -t '{}'".format(command, algo_params, self.data_path)

            try:
                returned_raw = sp.check_output(['bash', '-c', final_command], stderr = sp.STDOUT)
                mesure, returned, mesures_cols, mesures_vals = self.get_weka_learning_result(returned_raw, DEFAULT_MESURE)

                if mesure > max_mesure:
                    max_mesure = mesure
                    best_algo_params = algo_params
                    if self.weka.bagging_enabled:
                        best_bagging_params = bagging_params
                    best_mesures = [mesures_cols, mesures_vals]


                os.system('clear')
                print('Run {} => {} {}'.format(str(i).zfill(4), DEFAULT_MESURE, max_mesure), end = '\n\n')
                if self.weka.bagging_enabled:
                    print('Bagging Parameters : {}'.format(best_bagging_params))
                print('Algorithme Parameters : {}'.format(best_algo_params), end = '\n\n')
                print('Mesures :')
                for col, val in list(zip(best_mesures[0], best_mesures[1])):
                    print('{} : {}'.format(col, val))

                self.save_data(returned, mesure, algo_params, bagging_params)

            except Exception as e:
                lg.exception(e)


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

#!/usr/bin/python3
import os, sys, getopt, re, random
import subprocess as sp
import logging as lg

BASE_DIR = os.path.dirname(__file__) 
DEFAULT_WEKA_PATH = os.path.join( *[BASE_DIR, 'weka', 'weka.jar'] )

DEFAULT_ALGO_CLASS = "weka.classifiers.trees.J48"
BAGGING = "weka.classifiers.meta.Bagging"
DEFAULT_MESURE = 'ROC Area'
DONE_J48_PARAMS, DONE_BAGGING_PARAMS = [], []

def get_args(argv):
    data, output, weka_path, weka_class = "", "", "", ""

    try:
        opts, args = getopt.getopt(argv,"d:o:p:c",["data=","output=","weka-path","weka-class"])
    except getopt.GetoptError:
        print ('python3 learn.py [OPTIONS]\nReturns the best model for a certain dataset, uses Weka\n')
        print ('  -d, --data           dataset to use, file format must be compatibale with weka')
        print ('  -o, --output         model output file (must be .json, parent directory must exist)')
        print ('  -p, --weka-path      path to the weka.jar file')
        print ('  -c, --weka-class     weka class')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-d", "--data"]:
            data = arg
        elif opt in ["-o", "--output"]:
            output = arg
        elif opt in ["-p", "--weka-path"]:
            weka_path = arg
        elif opt in ["-c", "--weka-class"]:
            weka_class = arg

    return (data, output, weka_path, weka_class)


class Weka():
    __slots__ = ['weka_path', 'weka_class', 'bagging_enabled'] 
    
    def __init__(self, weka_path = None, weka_class = None, bagging_enabled = False):
        self.weka_path = self.check_weka_path(weka_path or DEFAULT_WEKA_PATH)
        self.weka_class = weka_class or DEFAULT_ALGO_CLASS
        self.bagging_enabled = bagging_enabled

    @classmethod
    def check_weka_path(cls, path):
        while not os.path.exists(path) :
            path = input("Weka could not be found, weka.jar path : ")
        return path


class Model():
    __slots__ = ['data_path', 'output_path'] 

    def __init__(self, data_path, output_path):
        self.data_path = self.check_data_path(data_path)
        # self.output_path = self.check_output_path(output_path)

    @classmethod
    def check_output_path(cls, path):
        while not os.path.exists( os.path.dirname(path) ) :
            path = input("output file directory doesn't exist, output path : ")
        return path

    @classmethod
    def check_data_path(cls, path):
        while not os.path.exists(path) :
            path = input("incorrect dataset path, dataset path : ")
        return path

    def get_weka_learning_result(self, raw_data, col = 'ROC Area'):
        try:
            all_data = raw_data.decode('utf-8')
            data = all_data[ all_data.index('Stratified cross-validation') : ]   #cval

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
            command = "{} {}".format(command, BAGGING)
        else:
            command = "{} {}".format(command, weka.weka_class)
        return command


    def learn(self, weka):
        max_mesure = 0
        best_algo_params = ''
        best_bagging_params = ''
        best_mesures = [[],[]]
        i = 0
        
        command = self.get_main_command(weka)

        while True:
            i += 1
            algo_params = self.get_j48_params()

            if weka.bagging_enabled:
                bagging_params = self.get_bagging_params()
                final_command = "{} {} -t '{}' -W {} -- {}".format(command, bagging_params, self.data_path, weka.weka_class, algo_params)
            else:
                final_command = "{} {} -t '{}'".format(command, algo_params, self.data_path)

            # print(params)
            # print(final_command)
            try:
                returned_raw = sp.check_output(['bash', '-c', final_command], stderr = sp.STDOUT)
                mesure, returned, mesures_cols, mesures_vals = self.get_weka_learning_result(returned_raw, DEFAULT_MESURE)
                if mesure > max_mesure:
                    max_mesure = mesure
                    best_algo_params = algo_params
                    
                    if weka.bagging_enabled:
                        best_bagging_params = bagging_params
                    best_mesures = [mesures_cols, mesures_vals]

                os.system('clear')
                print('Run {} => {} {}'.format(str(i).zfill(4), DEFAULT_MESURE, max_mesure), end = '\n\n')
                if weka.bagging_enabled:
                    print('Bagging Parameters : {}'.format(best_bagging_params))
                print('Algorithme Parameters : {}'.format(best_algo_params), end = '\n\n')
                print('Mesures :')
                for col, val in list(zip(best_mesures[0], best_mesures[1])):
                    print('{} : {}'.format(col, val))

            except Exception as e:
                lg.exception(e)

    def get_j48_params(self):
        global DONE_J48_PARAMS
        params = ''
        
        while params not in DONE_J48_PARAMS and params == '':
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

        DONE_J48_PARAMS += [params]

        return params

    def get_bagging_params(self):
        global DONE_BAGGING_PARAMS
        params = ''

        while params not in DONE_BAGGING_PARAMS and params == '':
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

        DONE_BAGGING_PARAMS += [params]
        return params


if __name__ == '__main__':
    ## getting information
    data, output, weka_path, weka_class = get_args(sys.argv[1:])

    ## intializing classes
    weka = Weka(weka_path, weka_class, True)
    model = Model(data, output)

    ## learning
    model.learn(weka)
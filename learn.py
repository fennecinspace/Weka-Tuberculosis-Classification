#!/usr/bin/python3
import os, json, sys, getopt, re
import subprocess as sp
import logging as lg

DEFAULT_ALGO_CLASS = "weka.classifiers.trees.J48"
DEFAULT_WEKA_PATH = "/opt/weka/weka.jar"

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
    __slots__ = ['weka_path', 'weka_class'] 
    
    def __init__(self, weka_path = None, weka_class = None):
        self.weka_path = self.check_weka_path(weka_path or DEFAULT_WEKA_PATH)
        self.weka_class = weka_class or DEFAULT_ALGO_CLASS

    @classmethod
    def check_weka_path(cls, path):
        while not os.path.exists(path) :
            path = input("Weka could not be found, weka.jar path : ")
        return path


class Model():
    __slots__ = ['data_path', 'output_path'] 

    def __init__(self, data_path, output_path):
        self.data_path = self.check_data_path(data_path)
        self.output_path = self.check_output_path(output_path)

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
        all_data = raw_data.decode('utf-8')
        data = all_data[ all_data.index('Stratified cross-validation') : ]   #cval
        print(data)

        columns_str = [line for line in data.splitlines() if line.find(col) != -1][0].strip()
        avgs_str = [line for line in data.splitlines() if line.find('Weighted Avg') != -1][0].strip()

        columns = re.findall('(\w+[ -]\w+|\w+)', columns_str)[:-1]
        avgs = re.findall('\d+\.\d+', avgs_str)
        print(columns)
        print(avgs)
        if columns.index(col) != -1:
            return float(avgs[columns.index(col)]), data
        else:
            print('Column ({}) could not be found !'.format(col))
            return 0, data

    def learn(self, command):
        ## adding dataset
        command = "{} {} '{}' -C 0.25 -M 2".format(command, "-t", self.data_path)
        print(command)
        # while True:
        try:
            returned_raw = sp.check_output(['bash', '-c', command], stderr = sp.STDOUT)
            mesure, returned = self.get_weka_learning_result(returned_raw, 'ROC Area')
            print(mesure)

        except Exception as e:
            lg.exception(e)





if __name__ == '__main__':
    ## getting information
    data, output, weka_path, weka_class = get_args(sys.argv[1:])

    ## intializing classes
    weka = Weka(weka_path, weka_class)
    model = Model(data, output)

    ## setting up main command
    command = "java -cp {} {}".format(weka.weka_path, weka.weka_class)

    ## learning
    model.learn(command)

    
# python3 learn.py -d $HOME"/Desktop/Machine Learning - M1 ISI/tp-projet/data/features/train/y_features_opt2_300_train_MODIFIED.csv" -o $HOME/Desktop/learn.json -p /opt/weka/weka.jar -c "weka.classifiers.trees.J48"
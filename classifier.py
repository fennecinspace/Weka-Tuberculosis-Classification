import os, sys, time
from learn import Model, Weka, get_args
# console size
rows, columns = os.popen('stty size', 'r').read().split()
rows, columns = int(rows), int(columns)

if __name__ == '__main__':
    try:
        ## getting information
        data, output, weka_path, weka_class, bypass_settings, bagging_enabled = get_args(sys.argv[1:])
        if '-h' in sys.argv or '--help' in sys.argv:
            exit()

        ## intializing classes
        weka = Weka(weka_path, weka_class, bagging_enabled)
        model = Model(weka, data, output)

        if not bypass_settings:
            ## Confirming Settings
            print('-' * columns)
            print('|Data     :', model.data_path)
            print('|Output   :', '{}/{}.json'.format(model.output_path, weka.weka_class))
            print('|Weka.jar :', weka.weka_path)
            print('|Class    :', weka.weka_class)
            print('|Bagging  :', end = ' ')
            if bagging_enabled : print('Enabled')
            else: print('Disabled')
            print('-' * columns)

            i = input('Are Settings Correct [Y][n] ? ')
            if i.lower().strip() == 'y':
                print('Starting...')
            else:
                print('Exiting.')
                exit()
        else:
            print('Starting...')

        ## learning
        model.learn()
    except:
        print('\b\b   ')
import os, sys, time
from learn import Model, Weka, get_args
# console size
rows, columns = os.popen('stty size', 'r').read().split()
rows, columns = int(rows), int(columns)

if __name__ == '__main__':
    try:
        ## getting information
        data, test, output, weka_path, weka_class, bypass_settings, bagging_enabled = get_args(sys.argv[1:])
        if '-h' in sys.argv or '--help' in sys.argv:
            exit()

        ## intializing classes
        weka = Weka(weka_path, weka_class, bagging_enabled)
        model = Model(weka, data, test, output)
        print(test)

        if not bypass_settings:
            ## Confirming Settings
            print('-' * columns)
            print('|Data     :', model.data_path)
            if model.test_data_path: print('|Test     :', model.test_data_path)
            print('|Output   :', model.output_path)
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


        if model.test_data_path: ## learning
            model.test()
        else: ## testing
            model.learn()
    except Exception as e:
        print(e)
        print('\b\b   ')
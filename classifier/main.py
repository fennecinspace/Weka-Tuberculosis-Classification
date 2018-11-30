import os, sys, time
from learn import Model, Weka, get_args

if __name__ == '__main__':
    ## getting information
    data, output, weka_path, weka_class = get_args(sys.argv[1:])

    ## intializing classes
    weka = Weka(weka_path, weka_class, True)
    model = Model(data, output)

    ## Confirming Settings
    print('data   :', model.data_path)
    print('output :', '{}/{}.json'.format(model.output_path, weka.weka_class))
    print('class  :', weka.weka_class)

    i = input('Are Settings Correct [Y][n] ? ')
    if i.lower().strip() == 'y':

        print('Starting...')
    else:
        print('Exiting.')
        exit()

    ## learning
    model.learn(weka)
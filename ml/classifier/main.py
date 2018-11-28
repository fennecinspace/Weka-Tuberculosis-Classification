from learn import Model, Weka
import os

BASE_DIR = os.path.abspath(os.path.dirname)


if __name__ == '__main__':
    ## getting information
    # data, cross, output, weka_path, weka_class = get_args(sys.argv[1:])
    data = os.path.join(*[BASE_DIR, 'datasets', 'features', 'train', 'y_features_opt2_300_train____________.csv'])
    weka_path = os.path.join(*[BASE_DIR, 'weka', 'weka.jar'])
    weka_class = "weka.classifiers.trees.J48"
    output = ""
    cross = None

    ## intializing classes
    weka = Weka(weka_path, weka_class, True)
    model = Model(data, cross, output)

    ## learning
    model.learn(weka)
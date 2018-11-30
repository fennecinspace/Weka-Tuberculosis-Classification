# Machine Learning Homework

For help:

    python3 learn.py -h


### Dev:
#### LOW / HIGH :

    python3 learn.py -d $(pwd)/datasets/features/train/y_features_opt2_300_train____________.csv \
        -v $(pwd)/datasets/features/val/y_features_opt2_300_val_HL.csv

    Run 0580 => ROC Area 0.633

    Bagging Parameters :  -P 68 -O -S 19 -store-out-of-bag-predictions -output-out-of-bag-complexity-statistics
    Algorithe Parameters :  -C 0.49180016676754446 -B -L -J

    Mesures :
    TP Rate : 0.654
    FP Rate : 0.348
    Precision : 0.654
    Recall : 0.654
    F-Measure : 0.654
    MCC : 0.306
    ROC Area : 0.633
    PRC Area : 0.614


#### 1, 2, 3, 4, 5 :

        python3 learn.py -d $(pwd)/datasets/features/train/y_features_opt2_300_train.csv

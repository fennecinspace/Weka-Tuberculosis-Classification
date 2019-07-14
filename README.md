# Machine Learning Homework

Tuberculosis Types Classification (5 types) using Weka

- input : 3D chest scans data for Tuberculosis patients
- output : a modele that classifies new chest data into one of 5 Tuberculosis types

#### GROUP: 
- Benkedadra Mohamed
- Youcefi Mohemmed Yassine
- Benkorreche Mohamed El Amine

### CLI:

- install requirements:

        sudo apt-get install python3

- For help:

        python3 classifier.py -h

### GUI:

- install requirements:
        
        sudo apt-get install python3 python3-pip gnome-terminal
        sudo pip3 install eel==0.9.7

- launch gui:

        cd gui
        python3 gui.py

### FILES:
- Train :
                
        $(pwd)/datasets/features/train/2classes/y_features_opt2_300_train.csv
        $(pwd)/datasets/features/train/2classes/z_features_opt2_300_train.csv
        $(pwd)/datasets/features/train/2classes/yz_features_opt2_600_train.csv
- Val :

        $(pwd)/datasets/features/val/2classes/y_features_opt2_300_val.csv
        $(pwd)/datasets/features/val/2classes/z_features_opt2_300_val.csv
        $(pwd)/datasets/features/val/2classes/yz_features_opt2_600_val.csv
- Train Val :
  
        $(pwd)/datasets/features/trainval/2classes/y_features_opt2_300_trainval.csv
        $(pwd)/datasets/features/trainval/2classes/z_features_opt2_300_trainval.csv
        $(pwd)/datasets/features/trainval/2classes/yz_features_opt2_300_trainval.csv

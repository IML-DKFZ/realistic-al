import os
import sys

"""
_______________________________________

-------- Does not work -------

Download Cars Dataset:
wget -O car_imgs.tgz http://ai.stanford.edu/\~jkrause/car196/car_ims.tgz

Unpack Cars Dataset:
tar zxvf car_imgs.tgz

--> Error and Data seems to be missing:
    Error Msg. tar:
        tar: Skipping to next header

-->
________________________________________ 

Stanford Cars Dataset

--------- Works ----------

Download Cars Annotations:
wget -O cars_annos.mat http://ai.stanford.edu/~jkrause/car196/cars_annos.mat

Download Cars Test Annotations:
wget -O cars_test_annos_withlabels.mat http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat

Download DevKit:
wget -O car_devkit.tgz http://ai.stanford.edu/\~jkrause/cars/car_devkit.tgz

Unpack DevKit:
tar zxvf car_devkit.tgz

Download Cars Training Set
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz

Download Cars Test Set
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz

Unpack Datsets:
tar zxvf cars_train.tgz
    --> Error msg:
        tar: Skipping to next header
        tar: Exiting with failure status due to previous errors
        
tar zxvf cars_test.tgz
"""

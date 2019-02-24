# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:10:57 2019

@author: DATAmadness
"""

############################################
###### TEST for skew autotransform##########

import pandas as pd
import numpy as np
from skew_autotransform import skew_autotransform

#Import test dataset - Boston huosing data
from sklearn.datasets import load_boston

exampleDF = pd.DataFrame(load_boston()['data'], columns = load_boston()['feature_names'].tolist())

transformedDF = skew_autotransform(exampleDF.copy(deep=True), plot = True, 
                                   exp = True, threshold = 0.7, exclude = ['B','LSTAT'])

print('Original average skewness value was %2.2f' %(np.mean(abs(exampleDF.skew()))))
print('Average skewness after transformation is %2.2f' %(np.mean(abs(transformedDF.skew()))))

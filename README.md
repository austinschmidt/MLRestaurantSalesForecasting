# Machine Learning Restaurant Sales Forecasting
Python code for the generation and testing of machine learning models, including the Temporal Fusion Transformer recurrent neural network model. Several Jupyter Notebook files are available with different test suites to cover feature testing, model tuning, one-day forecast tests, and one-week forecast tests. Also included are three datasets. An actual sales dataset and two derivative datasets made from daily and weekly sales differencing.

To render the Jupyter Notebook pages reliably without dowloading, copy and paste the URL for this repository at: https://nbviewer.jupyter.org/
The files will be rendered for quick view.


## The supplied dataset is organized as follows:
There are two files holding data.

### (1) RestaurantDataVets_All_2to5.csv
 
The raw input data has been preprocessed to have many features with the target being '2to5' which is the amount of sales generated between 2:00 and 5:59 PM on that day. The Jupyter Notebook code handles the addition of lookback features, one-hot encoding, and data scaling. The first 7 days are used to generate statistical measurements, and must be thrown out before training. 

### (2) RestaurantDataVets_All_2to5_Differenced.csv

The differenced dataset has additional features and targets to benefit from. The targets are: '2to5', 'DailyDifference', 'WeeklyDifference', and 'DiffDifference', and care should be made to drop targets not being used before training. While '2to5' are actual sales, daily difference is the difference in sales between today and yesterday, weekly difference is the difference between today and last week, and diff difference applies both differencing techniques. The Jupyter Notebook code handles the addition of lookback features, one-hot encoding, and data scaling. The first 14 days are used to generate statistical measurements, and must be thrown out before training. 

The raw dataset may be aquired by directly reaching out to the author at sbaustin@uno.edu.

## The Jupyter Notebook files are organized as follows:
There are 5 Jupyter Notebook files holding training/testing procedures for a large survey of models. Files should be run in cell order to ensure desired outputs.

### (1) Scikit Feature Selection.ipynb

### (2) Keras Feature Selection.ipynb

### (3) TransformerTuning.ipynb 

### (4) FinalTestingOneDay.ipynb

### (5) FinalTestingOneWeek.ipynb

**Note** The following features were removed from one-week forecasting:  DailyAvg, DailyBusyness, and AvgDailyDiff due to concerns of improper information gain, however this was a mistake and these features really can be used. Luckily they rank low on feature importance, so it is likely this hurt performance very little. 

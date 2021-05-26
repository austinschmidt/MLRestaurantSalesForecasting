# Machine Learning Restaurant Sales Forecasting
Python code for the generation and testing of machine learning models, including the Temporal Fusion Transformer recurrent neural network model. Several Jupyter Notebook files are available with different test suites to cover feature testing, model tuning, one-day forecast tests, and one-week forecast tests. Also included are three datasets. An actual sales dataset and two derivative datasets made from daily and weekly sales differencing.

To render the Jupyter Notebook pages reliably without dowloading, copy and paste the URL for this repository at: https://nbviewer.jupyter.org/
The files will be rendered for quick view.


## The supplied dataset is organized as follows:
There are two files holding data.

(1) RestaurantDataVets_All_2to5.csv
 
The raw input data has been preprocessed to have many features with the target being '2to5' which is the amount of sales generated between 2:00 and 5:59 PM on that day. The Jupyter Notebook code handles the addition of lookback features, one-hot encoding, and data scaling. The first 7 days are used to generate statistical measurements, and must be thrown out before training. 

(2) RestaurantDataVets_All_2to5_Differenced.csv


The raw dataset may be aquired by directly reaching out to the author at sbaustin@uno.edu.

## The Jupyter Notebook files are organized as follows:





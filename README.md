# Atelier 1 — Deep Learning
Asmaa Atouailaa

This lab uses PyTorch MLPs for two tasks: predicting a stock price and identifying a machine failure type.

## Running the notebooks
Open the notebooks in Google Colab and run the cells in order. Upload the requested CSV when prompted. The installation cells add skorch and imbalanced-learn. The notebooks also use pandas, NumPy, scikit-learn, PyTorch, matplotlib and seaborn. Outputs from the completed experiments are included. The CSV files are not included.

- `Atelier1_Regression_Asmaa.ipynb`: upload `prices-split-adjusted.csv` from [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse).
- `Atelier1_Classification_Asmaa.ipynb`: upload `predictive_maintenance.csv` from [Predictive maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).

## Part 1: Regression
The target is the next trading day's closing price for the same company. Inputs are today's open, close, low, high and volume, available after today's close. The data contains 851,264 rows and 501 symbols from 2010 to 2016.

Training uses targets before 2015, validation covers 2015 and testing covers 2016. Pairs crossing split boundaries are excluded. Scaling uses training statistics only.

GridSearchCV compares 16 configurations on one chronological validation split. It selects Adam, learning rate 0.001, 20 epochs and hidden layers of 32 and 16 neurons.

| Model | Validation MAE ($) | Validation RMSE ($) |
|---|---:|---:|
| First network | 1.215386 | 2.544084 |
| Tuned network | 1.072572 | 2.443672 |
| Weight decay | 1.110217 | 2.512901 |
| Dropout | 1.791637 | 4.879515 |

Tuning improves the first network. Dropout 0.1 and weight decay 0.0001 do not improve the tuned result.

| Method | Test MAE ($) | Test RMSE ($) | Test R² |
|---|---:|---:|---:|
| Tuned network | 1.076454 | 2.512573 | 0.999409 |
| Today's closing price | 0.948386 | 2.142949 | 0.999570 |

The simple baseline wins. The high R² should not be interpreted as forecasting accuracy. The loss curves fluctuate, with test error above training error. A separate fixed fit records training/test curves. R² replaces the assignment's ambiguous regression accuracy curve; it is clearly labelled as a regression score. Any teacher-specific accuracy definition would need clarification.

## Part 2: Classification
The target is Failure Type. Inputs are five measurements and product type. Target is excluded because it reveals failure status; identifiers are also excluded.

Original observations are split 60/20/20 with stratification. Imputation, scaling and encoding use training data only. SMOTENC balances the training set to 5,791 examples per category. Validation and test sets retain their original distribution.

A small GridSearchCV compares four grouped configurations on one validation split, using macro-F1. The selected settings are Adam, 0.001, 15 epochs and hidden layers of 64 and 32 neurons.

| Model | Validation macro-F1 |
|---|---:|
| First model | 0.423160 |
| Tuned model | 0.500670 |
| Dropout | 0.445607 |
| Weight decay | 0.504783 |

Weight decay is selected using validation only. Its advantage is small and has not been confirmed across repeated runs.

| Method | Test accuracy | Test macro recall | Test macro-F1 |
|---|---:|---:|---:|
| Always No Failure | 0.965 | 0.166667 | 0.163698 |
| Selected configuration, reporting fit | 0.905 | 0.723857 | 0.502207 |

Training/test loss generally decreases and accuracy rises. The reporting model's training accuracy is 0.921333 and macro-F1 is 0.564369. It detects 23/23 heat failures, 17/19 power failures and 6/9 tool wear failures on the test set. It misses all three random failures and labels 180 healthy cases as failures.

## What I learned
The regression experiment showed why a simple baseline matters: the network did not beat today's price as a prediction. The classification experiment showed why high accuracy can hide poor failure detection. Per-class recall and F1 gave a more useful picture.

I also learned how to build MLPs, separate training from evaluation, tune settings and test regularization. Test curves were recorded only after fixing the configuration and training duration. Reporting fits are separate from earlier fits and may give slightly different results. The search was limited, and the rare categories contain very few real examples. Synthetic examples do not remove that limitation, and no unbalanced-data comparison was run to establish an augmentation benefit.

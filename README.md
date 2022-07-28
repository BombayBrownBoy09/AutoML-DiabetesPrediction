# AutoML-Diabetes Prediction
Classifying Diabetes cases using  FLAML, to train, evaluate, and deploy a model

## FLAML AutoML

FLAML (Fast and Lightweight AutoML) is a lightweight and efficient library implemented in Python and it has a scikit-learn style API. It allows the users to not worry about selecting the right machine learning algorithms or hyperparameters for each algorithm.

[Documentation](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/)

```bash
./model_training.py
```

### Results

```bash
/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_sag.py:354: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  ConvergenceWarning,
[flaml.automl: 07-28 22:02:47] {3179} INFO -  at 50.1s,	estimator lrl1's best error=0.3052,	best estimator lgbm's best error=0.2305
[flaml.automl: 07-28 22:02:47] {3439} INFO - retrain lgbm for 0.0s
[flaml.automl: 07-28 22:02:47] {3444} INFO - retrained model: LGBMClassifier(colsample_bytree=0.9498119875710125,
               learning_rate=0.7302320185843355, max_bin=127,
               min_child_samples=33, n_estimators=4, num_leaves=6,
               reg_alpha=0.0009765625, reg_lambda=12.40456319727597,
               verbose=-1)
[flaml.automl: 07-28 22:02:47] {2722} INFO - fit succeeded
[flaml.automl: 07-28 22:02:47] {2724} INFO - Time taken to find the best model: 34.293384313583374
Best Model <flaml.model.LGBMEstimator object at 0x7f361bc202d0>
Best hyperparmeter config: {'n_estimators': 4, 'num_leaves': 6, 'min_child_samples': 33, 'learning_rate': 0.7302320185843355, 'log_max_bin': 7, 'colsample_bytree': 0.9498119875710125, 'reg_alpha': 0.0009765625, 'reg_lambda': 12.40456319727597}
Best accuracy on validation data: 0.7695
Training duration of best run: 0.01133 s
```

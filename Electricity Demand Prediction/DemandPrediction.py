import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from CleaningSortingManipulating import df_concatenated

data = df_concatenated[["ND", "SETTLEMENT_PERIOD", "NON_BM_STOR", "PUMP_STORAGE_PUMPING",
                        "EMBEDDED_RENEWABLE_GENERATION", "EMBEDDED_RENEWABLE_CAPACITY",
                        "TOTAL_FLOW"]]

# label - the variable we want to predict - grade at end of term 3
predict = 'ND'

# training data - we drop the column we want to predict
X = np.array(data.drop([predict], axis = 1))
# actual ND values
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.1)


# skip whole training process, only need to 'uncomment' for first run of file
# file has already been run with this once and saves the best model
best = 0
for _ in range(10):
    # train vars to train model, test vars to test accuracy - testing are 'unseen' data
    # test_size = ___ takes that proportion of the data as the test set
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.01)

    #find non-linear relationship
    model = RandomForestRegressor(n_estimators=4, max_features=3)
    # finds best fit line
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    # changes every time we train the model so different result every time
    print(accuracy)

    # only save new model if current score better than already seen score
    if accuracy > best:
        best = accuracy
        # writes the file if it doesn't already exist
        with open("demandmodel.pickle", "wb") as f:
            # saves the model into the file f
            pickle.dump(model, f)

# read in the pickle file after creating it
pickle_in = open("demandmodel.pickle", "rb")
# load pickle into linear model
model = pickle.load(pickle_in)


predictions = model.predict(x_test) # note same length as x_test
print("Prediction vs input vs actual for 10 examples: \n")
for i in range(10):
    print(predictions[i], x_test[i], y_test[i])

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, predictions)), '.3f'))
print("\nRMSE: ", rmse)
# rmse is pretty good considering values can range from 0-50000

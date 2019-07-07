import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
msqs = []
coeffd = []
intercepts = []
mmres = []
for i in range(1,57):
    file_name = "/home/zahed/Desktop/MLAssignment2/{}.csv".format(i)
    df = pd.read_csv(file_name, header=None, index_col=False)
    df.dropna(axis=0)
    df.dropna(axis=1)
    normalized_df = (df-df.min())/(df.max()-df.min())
    normalized_df.fillna(0, inplace=True)
    x = normalized_df.iloc[:, 0:21].values
    y = normalized_df.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_preds = regressor.predict(x_test)

    y_preds_rounded = []
    for k in range(y_preds.size):
        y_preds_rounded.append(y[abs(y-y_preds[k]).argmin()])

    y_test_mod = y_test + 0.01

    msqs.append(metrics.mean_squared_error(y_test, y_preds_rounded))
    coeffd.append(regressor.coef_)
    intercepts.append(regressor.intercept_)

    mres_df = abs(y_preds_rounded - y_test)/y_test_mod
    mmres.append(np.mean(mres_df))
    
    pd.DataFrame(msqs).to_excel(excel_writer="/home/<user>/Desktop/MLAssignment2/Rounding_to_closest_y_value/MeanSq.xlsx", header=None, index=True)
    pd.DataFrame(coeffd).to_excel(excel_writer="/home/<user>/Desktop/MLAssignment2/Rounding_to_closest_y_value/Coefficients.xlsx", header=None, index=True)
    pd.DataFrame(intercepts).to_excel(excel_writer="/home/<user>/Desktop/MLAssignment2/Rounding_to_closest_y_value/Intercepts.xlsx", header=None, index=True)
    pd.DataFrame(mmres).to_excel(excel_writer="/home/<user>/Desktop/MLAssignment2/Rounding_to_closest_y_value/MeanMagRelError.xlsx", header=None, index=True)

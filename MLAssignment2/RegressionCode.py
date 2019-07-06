import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
msqs = []
coeffd = []
intercepts = []
mmres = []
for i in range(1,57):
    file_name = "/home/<user_name>/Desktop/MLAssignment2/{}.csv".format(i)
    df = pd.read_csv(file_name, header=None, index_col=False)
    df.dropna(axis=0)
    df.dropna(axis=1)
    x = df.iloc[:, 0:21].values
    y = df.iloc[:, -1].values
    normalized_df = (x-x.min())/(x.max()-x.min())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_preds = regressor.predict(x_test)

    y_test_mod = y_test + 0.01

    msqs.append(metrics.mean_squared_error(y_test, y_preds))
    coeffd.append(regressor.coef_)
    intercepts.append(regressor.intercept_)

    mres_df = abs(y_preds - y_test)/y_test_mod
    mmres.append(np.mean(mres_df))
    
    pd.DataFrame(msqs).to_excel(excel_writer="/home/<user_name>/Desktop/MLAssignment2/MeanSq.xlsx", header=None, index=True)
    pd.DataFrame(coeffd).to_excel(excel_writer="/home/<user_name>/Desktop/MLAssignment2/Coefficients.xlsx", header=None, index=True)
    pd.DataFrame(intercepts).to_excel(excel_writer="/home/<user_name>/Desktop/MLAssignment2/Intercepts.xlsx", header=None, index=True)
    pd.DataFrame(mmres).to_excel(excel_writer="/home/<user_name>/Desktop/MLAssignment2/MeanMagRelError.xlsx", header=None, index=True)

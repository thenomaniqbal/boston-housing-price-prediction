# ---------------------------
# organize imports
# ---------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import warnings
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------
# library specific options
# ---------------------------
pd.options.display.float_format = '{:,.2f}'.format
sns.set(color_codes=True)


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------
# analyze the dataset
# ---------------------------
def analyze_dataset(dataset):
    print("[INFO] keys : {}".format(dataset.keys()))

    print("[INFO] features shape : {}".format(dataset.data.shape))
    print("[INFO] target shape   : {}".format(dataset.target.shape))

    print("[INFO] feature names")
    print(dataset.feature_names)

    print("[INFO] dataset summary")
    print(dataset.DESCR)

    df = pd.DataFrame(dataset.data)
    print("[INFO] df type : {}".format(type(df)))
    print("[INFO] df shape: {}".format(df.shape))
    print(df.head())

    df.columns = dataset.feature_names
    print(df.head())

    df["PRICE"] = dataset.target
    print(df.head())

    print("[INFO] dataset datatypes")
    print(df.dtypes)

    print("[INFO] dataset statistical summary")
    print(df.describe())

    # correlation between attributes
    print("PEARSON CORRELATION")
    print(df.corr(method="pearson"))
    sns.heatmap(df.corr(method="pearson"))
    plt.savefig("heatmap_pearson.png")
    plt.clf()
    plt.close()

    print("SPEARMAN CORRELATION")
    print(df.corr(method="spearman"))
    sns.heatmap(df.corr(method="spearman"))
    plt.savefig("heatmap_spearman.png")
    plt.clf()
    plt.close()

    print("KENDALL CORRELATION")
    print(df.corr(method="kendall"))
    sns.heatmap(df.corr(method="kendall"))
    plt.savefig("heatmap_kendall.png")
    plt.clf()
    plt.close()

    # show missing values
    print(pd.isnull(df).any())

    file_report = "boston_housing.txt"
    with open(file_report, "w") as f:
        f.write("Features shape : {}".format(df.drop("PRICE", axis=1).shape))
        f.write("\n")

        f.write("Target shape   : {}".format(df["PRICE"].shape))
        f.write("\n")

        f.write("\nColumn names")
        f.write("\n")
        f.write(str(df.columns))
        f.write("\n")

        f.write("\nStatistical summary")
        f.write("\n")
        f.write(str(df.describe()))
        f.write("\n")

        f.write("\nDatatypes")
        f.write("\n")
        f.write(str(df.dtypes))
        f.write("\n")

        f.write("\nPEARSON correlation")
        f.write("\n")
        f.write(str(df.corr(method="pearson")))
        f.write("\n")

        f.write("\nSPEARMAN correlation")
        f.write("\n")
        f.write(str(df.corr(method="spearman")))
        f.write("\n")

        f.write("\nKENDALL correlation")
        f.write("\n")
        f.write(str(df.corr(method="kendall")))

        f.write("\nMissing Values")
        f.write("\n")
        f.write(str(pd.isnull(df).any()))

    return df


# ---------------------------
# visualize the dataset
# ---------------------------
def visualize_dataset(df):
    colors = ["y", "b", "g", "r"]

    cols = list(df.columns.values)

    if not os.path.exists("plots/univariate/box"):
        os.makedirs("plots/univariate/box")

    if not os.path.exists("plots/univariate/density"):
        os.makedirs("plots/univariate/density")

    # draw a boxplot with vertical orientation
    for i, col in enumerate(cols):
        sns.boxplot(df[col], color=random.choice(colors), orient="v")
        plt.savefig("plots/univariate/box/box_" + str(i) + ".png")
        plt.clf()
        plt.close()

    # draw a histogram and fit a kernel density estimate (KDE)
    for i, col in enumerate(cols):
        sns.distplot(df[col], color=random.choice(colors))
        plt.savefig("plots/univariate/density/density_" + str(i) + ".png")
        plt.clf()
        plt.close()

    if not os.path.exists("plots/multivariate"):
        os.makedirs("plots/multivariate")

    # bivariate plot between target and feature
    for i, col in enumerate(cols):
        if (i == len(cols) - 1):
            pass
        else:
            sns.jointplot(x=col, y="PRICE", data=df);
            plt.savefig("plots/multivariate/target_vs_" + str(i) + ".png")
            plt.clf()
            plt.close()

    # pairplot
    sns.pairplot(df)
    plt.savefig("plots/pairplot.png")
    plt.clf()
    plt.close()


# ---------------------------
# train the model
# ---------------------------
def train_model(df, dataset):
    X = df.drop("PRICE", axis=1)
    Y = df["PRICE"]
    print(X.shape)
    print(Y.shape)

    scaler = MinMaxScaler().fit(X)
    scaled_X = scaler.transform(X)

    seed = 9
    test_size = 0.20

    X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size=test_size, random_state=seed)

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    # user variables to tune
    folds = 10
    metric = "neg_mean_squared_error"

    # hold different regression models in a single dictionary
    models = {}
    models["Linear"] = LinearRegression()
    models["Lasso"] = Lasso()
    models["ElasticNet"] = ElasticNet()
    models["KNN"] = KNeighborsRegressor()
    models["DecisionTree"] = DecisionTreeRegressor()
    models["SVR"] = SVR()
    models["AdaBoost"] = AdaBoostRegressor()
    models["GradientBoost"] = GradientBoostingRegressor()
    models["RandomForest"] = RandomForestRegressor()
    models["ExtraTrees"] = ExtraTreesRegressor()

    # 10-fold cross validation for each model
    model_results = []
    model_names = []
    for model_name in models:
        model = models[model_name]
        k_fold = KFold(n_splits=folds, shuffle=True, random_state=seed)
        results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)

        model_results.append(results)
        model_names.append(model_name)
        print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

    # box-whisker plot to compare regression models
    figure = plt.figure()
    figure.suptitle('Regression models comparison')
    axis = figure.add_subplot(111)
    plt.boxplot(model_results)
    axis.set_xticklabels(model_names, rotation=45, ha="right")
    axis.set_ylabel("Mean Squared Error (MSE)")
    plt.margins(0.05, 0.1)
    plt.savefig("model_mse_scores.png")
    plt.clf()
    plt.close()

    # create and fit the best regression model
    best_model = GradientBoostingRegressor(random_state=seed)
    best_model.fit(X_train, Y_train)

    # make predictions using the model
    predictions = best_model.predict(X_test)
    print("[INFO] MSE : {}".format(round(mean_squared_error(Y_test, predictions), 3)))

    # plot between predictions and Y_test
    x_axis = np.array(range(0, predictions.shape[0]))
    plt.plot(x_axis, predictions, linestyle="--", marker="o", alpha=0.7, color='r', label="predictions")
    plt.plot(x_axis, Y_test, linestyle="--", marker="o", alpha=0.7, color='g', label="Y_test")
    plt.xlabel('Row number')
    plt.ylabel('PRICE')
    plt.title('Predictions vs Y_test')
    plt.legend(loc='lower right')
    plt.savefig("predictions_vs_ytest.png")
    plt.clf()
    plt.close()

    # plot model's feature importance
    feature_importance = best_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, dataset.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.savefig("feature_importance.png")
    plt.clf()
    plt.close()


# --------------------------
# MAIN FUNCTION
# --------------------------
if __name__ == '__main__':
    dataset = load_boston()

    df = analyze_dataset(dataset)
    visualize_dataset(df)
    train_model(df, dataset)
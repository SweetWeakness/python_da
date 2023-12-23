import argparse
import pandas
import pickle
from sklearn.linear_model import LinearRegression, LassoLars

from utils import *
import logging
logging.basicConfig(level=logging.INFO)


def loadData():
    def loadFromPickle(path):
        with open(path, "rb") as fp:
            result = pickle.load(fp)
        return result

    X_train = loadFromPickle(f"{DATA_PATH}/X_train.pickle")
    X_dev = loadFromPickle(f"{DATA_PATH}/X_dev.pickle")
    y_train = loadFromPickle(f"{DATA_PATH}/y_train.pickle")
    y_dev = loadFromPickle(f"{DATA_PATH}/y_dev.pickle")
    return X_train, X_dev, y_train, y_dev


def loadModels():
    try:
        with open(f"{MODELS_PATH}/model1.pickle", "rb") as fp:
            model1 = pickle.load(fp)
    except FileNotFoundError:
        model1 = LinearRegression()
    try:
        with open(f"{MODELS_PATH}/model2.pickle", "rb") as fp:
            model2 = pickle.load(fp)
    except FileNotFoundError:
        model2 = LassoLars(normalize=True)
    
    return model1, model2


def saveModels(model1, model2):
    with open(f"{MODELS_PATH}/model1.pickle", "wb") as fp:
        pickle.dump(model1, fp)
    with open(f"{MODELS_PATH}/model2.pickle", "wb") as fp:
        pickle.dump(model2, fp)


def trainModel(model1, model2, X_train, y_train):
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    y = model2.predict(X_train)
    report1 = model1.score(X_train, y_train)
    report2 = model2.score(X_train, y_train)
    logging.info(f"Regression score = {report1}")
    logging.info(f"LassoLars score = {report2}")
    with open(f"{LOG_PATH}/train_log.txt", "w") as outp:
        outp.write(f"regression_score: {report1}\nlassolars_score: {report2}")
    return model1, model2


def saveResult(model1, model2, X_dev, y_dev):
    report1 = model1.score(X_dev, y_dev)
    report2 = model2.score(X_dev, y_dev)
    logging.info(f"Saving results, reg_score = {report1}, lassolars_score = {report2}")
    with open(f"{RESULTS_PATH}/dev_result.txt", "w") as outp:
        outp.write(f"regression_score: {report1}\nlassolars_score: {report2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="default")
    args = parser.parse_args()
    run_name = args.run_name

    global RUN_NAME, DATA_PATH, MODELS_PATH, RESULTS_PATH, LOG_PATH

    RUN_NAME = run_name
    DATA_PATH = f"./data/{run_name}"
    MODELS_PATH = f"./models/{run_name}"
    RESULTS_PATH = f"./results/{run_name}"
    LOG_PATH = f"./log/{run_name}"

    make_dir(f"{MODELS_PATH}")
    make_dir(f"{LOG_PATH}")
    make_dir(f"{RESULTS_PATH}")

    X_train, X_dev, y_train, y_dev = loadData()
    model1, model2 = loadModels()
    model1, model2 = trainModel(model1, model2, X_train, y_train)
    saveModels(model1, model2)
    saveResult(model1, model2, X_dev, y_dev)


if __name__ == "__main__":
    main()

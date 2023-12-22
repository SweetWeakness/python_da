import argparse
import pandas
import pickle
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

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


def loadModel():
    try:
        with open(f"{MODELS_PATH}/model.pickle", "rb") as fp:
            model = pickle.load(fp)
    except FileNotFoundError:
        model = LinearRegression()
    return model


def saveModel(model):
    with open(f"{MODELS_PATH}/model.pickle", "wb") as fp:
        pickle.dump(model, fp)


def trainModel(model, X_train, y_train):
    model.fit(X_train, y_train)
    report = model.score(X_train, y_train)
    logging.info(f"Regression report, score = {report}")
    with open(f"{LOG_PATH}/train_log.txt", "w") as outp:
        outp.write(f"score: {report}")
    return model


def saveResult(model, X_dev, y_dev):
    report = model.score(X_dev, y_dev)
    logging.info(f"Saving results, score = {report}")
    with open(f"{RESULTS_PATH}/dev_result.txt", "w") as outp:
        outp.write(f"score: {report}")


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
    model = loadModel()
    model = trainModel(model, X_train, y_train)
    saveModel(model)
    saveResult(model, X_dev, y_dev)


if __name__ == "__main__":
    main()

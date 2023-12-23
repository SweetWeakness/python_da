import argparse
import os
import pickle
import pandas

from utils import MODELS_PATH, make_dir,RESULTS_PATH,LOG_PATH,RUN_NAME,DATA_PATH


def load_data():
    def unpickle(path):
        with open(path, "rb") as fp:
            result = pickle.load(fp)
        return result
    
    X_train = unpickle(f"{DATA_PATH}/X_train.pickle")
    X_dev = unpickle(f"{DATA_PATH}/X_dev.pickle")
    X_test = unpickle(f"{DATA_PATH}/X_test.pickle")
    y_train = unpickle(f"{DATA_PATH}/y_train.pickle")
    y_dev = unpickle(f"{DATA_PATH}/y_dev.pickle")
    y_test = unpickle(f"{DATA_PATH}/y_test.pickle")
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def load_models():
    with open(f"{MODELS_PATH}/model1.pickle", "rb") as fp:
        model1 = pickle.load(fp)
    with open(f"{MODELS_PATH}/model2.pickle", "rb") as fp:
        model2 = pickle.load(fp)
    return model1, model2


def predict_and_save(model1, model2, X, y):
    report1 = model1.score(X, y)
    report2 = model2.score(X, y)
    with open(f"{RESULTS_PATH}/result.txt", "w") as outp:
        outp.write(f"reg_score: {report1}\nlassolars_score: {report2}")


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

    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()
    model1, model2 = load_models()
    predict_and_save(model1, model2, X_test, y_test)


if __name__ == "__main__":
    main()

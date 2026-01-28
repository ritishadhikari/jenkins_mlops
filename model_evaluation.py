import joblib
from sklearn.metrics import accuracy_score, classification_report
import logging

logger=logging.getLogger(name=__name__)
logger.setLevel(level=logging.INFO)

formatter=logging.Formatter(fmt="%(asctime)s:%(levelname)s:%(message)s")
streamHandler=logging.StreamHandler()
streamHandler.setFormatter(fmt=formatter)


def load_preprocessed_data(file_path):
    return joblib.load(file_path)

def load_model(file_path):
    return joblib.load(file_path)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_preprocessed_data(file_path="artifacts/preprocessed_data.pkl")
    model = load_model(file_path="artifacts/model.pkl")
    accuracy, report = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    logger.info(msg=f"Accuracy: {accuracy}")
    logger.info(msg=f"Classification Report: \n {report}")
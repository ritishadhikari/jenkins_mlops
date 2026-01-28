from sklearn.ensemble import RandomForestClassifier
import joblib

import logging

logger=logging.getLogger(name=__name__)
logger.setLevel(level=logging.INFO)

formatter=logging.Formatter(fmt="%(asctime)s:%(levelname)s:%(message)s")
streamHandler=logging.StreamHandler()
streamHandler.setFormatter(fmt=formatter)

logger.addHandler(hdlr=streamHandler)

def load_preprocessed_data(file_path):
    return joblib.load(file_path)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    joblib.dump(model, file_path)
    logger.info(msg=f"Saved Preprocessed data in the filepath: {file_path}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_preprocessed_data("artifacts/preprocessed_data.pkl")
    model = train_model(X_train, y_train)
    save_model(model, "artifacts/model.pkl")

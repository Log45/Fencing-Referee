import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier

from classifier_data import generate_training_data

class PointDecider:
    def __init__(self, path = None, n_estimators=1):
        if path is not None:
            self.load(path)
        else:
            self.model = RandomForestClassifier(n_estimators=n_estimators)

    def save(self, path = "trained_models/point_decider/point_decider.pkl"):
        joblib.dump(self.model, path)
        
    def load(self, path = "trained_models/point_decider/point_decider.pkl"):    
        self.model = joblib.load(path)
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.print_metrics(X_test, y_test)
        self.save()
        
    def print_metrics(self, X, y):
        print(sklearn.metrics.classification_report(y, self.model.predict(X)))
    
    def predict(self, X):
        return self.model.predict(X)
    
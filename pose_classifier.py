import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier

from classifier_data import generate_training_data

class PoseClassifier:
    def __init__(self, n_estimators=1):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def save(self, path = "trained_models/pose_classifier/pose_classifier.pkl"):
        joblib.dump(self.model, path)
        
    def load(self, path = "trained_models/pose_classifier/pose_classifier.pkl"):    
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

def main(): 
    data, classes = generate_training_data()
    
    X = data[:, :-1]
    y = data[:, -1]
    
    clf = PoseClassifier(n_estimators=100)
    clf.train(X, y)
    
if __name__ == "__main__":
    main()
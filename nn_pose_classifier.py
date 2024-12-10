import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from classifier_data import generate_training_data

# Define the neural network
class SimpleNNClassifier(nn.Module):
    def __init__(self, path = None, input_size=26, num_classes=4, device="cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
        super(SimpleNNClassifier, self).__init__()
        self.device = device
        if path is not None:
            self.load(path)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),          # Fully connected layer with 128 units
                nn.ReLU(),                           # Non-linear activation
                nn.BatchNorm1d(128),                 # Batch normalization
                nn.Dropout(0.3),                     # Dropout for regularization
                nn.Linear(128, 64),                  # Second fully connected layer
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),          # Output layer with num_classes units
            )
        self.model.to(self.device)
    
    def forward(self, x: torch.Tensor):
        return self.model(x.to(self.device))
    
    # Calculate accuracy (a classification metric)
    @staticmethod
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100 
        return acc
    
    def predict(self, x):
        return torch.softmax(self.forward(x.to(self.device)), dim=1).argmax(dim=1)
    
    def predict_probs(self, x):
        return torch.softmax(self.forward(x.to(self.device)), dim=1)
    
    def fit(self, X, y, epochs=100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters())
        
        for epoch in range(epochs):
            self.model.train()
            
            y_logits = self.forward(X_train)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            
            loss = criterion(y_logits, y_train)
            acc = self.accuracy_fn(y_train, y_pred)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            self.model.eval()
            with torch.inference_mode():
                test_logits = self.forward(X_test)
                test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                test_loss = criterion(test_logits, y_test)
                test_acc = self.accuracy_fn(y_test, test_pred)            
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f},  Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
        
        print(classification_report(y_test.cpu().numpy(), self.predict(X_test).cpu().numpy()))
    
    def load(self, path):
        self.model = torch.load(path)

def main():
    # Load the data
    data, classes = generate_training_data()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    X = torch.tensor(data[:, :-1], dtype=torch.float32).to(device)
    y = torch.tensor(data[:, -1], dtype=torch.long).to(device)
    
    # 4. Plot data
    # plt.figure(figsize=(10, 7))
    # plt.scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c=y.cpu().numpy(), cmap=plt.cm.RdYlBu);
    # plt.show()
    
    # Initialize the model
    clf = SimpleNNClassifier(input_size=X.shape[1], num_classes=len(classes), device=device)
    
    # Train the model
    clf.fit(X, y, epochs=2000)
    
    # Save the model
    torch.save(clf, "trained_models/pose_classifier/pose_classifier.pth")
    
if __name__ == "__main__":
    main()
    
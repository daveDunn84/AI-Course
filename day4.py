import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset and split it into training and testing sets

data = pd.read_csv('spam.csv')

X = data.drop('spam', axis=1) # drops the column 'spam'

y = data['spam'] # pulls column into this variable (spam indicator)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model to classify models as spam / not spam
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(X_test)

# Evaluate the model using accuracy, confusion matrix, precision, recall and f1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Add labels to the printed results for clarity
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# Visualize the confusion matrix using Seaborns heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()
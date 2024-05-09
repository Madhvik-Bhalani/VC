# Importing the libraries
from data_preprocessing import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score ,precision_score, recall_score, f1_score
import xgboost
import seaborn as sns
import joblib

# Training the XG Boost  model on the Training set
classifier = xgboost.XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # [Predicted_Result,Actual_Result]

# save the classification model as a pickle file
joblib.dump(classifier, "models/pkl_files/XGBoost.pkl")

# Predicting a new result
new_result=classifier.predict(sc.transform([[1,140,70,41,168,30.5,0.53,25]]))

if(new_result==1):
    print("This Person is diabetic")
else:
    print("This person is not diabetic")

# Making the Confusion Matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac_score=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)

print(cm)
print("Accuracy: {:.2f}%".format(ac_score * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))

# Plot confusion matrix as heatmap

class_labels = ['0', '1']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('Confusion Matrix(XGBoost)')
plt.show()





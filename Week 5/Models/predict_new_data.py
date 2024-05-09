from sklearn.preprocessing import StandardScaler
import joblib


def predict_new_data(data):
    #apply feature scalling to data
    sc = StandardScaler()
    sc.fit_transform(data)
    
    result_set = []
    
    # load model
    XGboost = joblib.load("Models/pkl_files/XGBoost.pkl")
    DecisionTree_Classifier = joblib.load('Models/pkl_files/DecisionTree.pkl')
    KernalSVM = joblib.load('Models/pkl_files/KernalSVM.pkl')
    KNeighborsClassifier = joblib.load('Models/pkl_files/K-Neighbors.pkl')
    LogisticRegression = joblib.load('Models/pkl_files/LogisticRegression.pkl')
    naiveBayes = joblib.load('Models/pkl_files/naiveBayes.pkl')
    RandomForestClassifier = joblib.load('Models/pkl_files/RandomForest.pkl')
    SVM = joblib.load('Models/pkl_files/Svm.pkl')
    
    #predict data and append in result set
    result_set.append(["XGBoost", XGboost.predict(data)])
    result_set.append(["DecisionTree Classifier", DecisionTree_Classifier.predict(data)])
    result_set.append(["KernalSVM", KernalSVM.predict(data)])
    result_set.append(["K-Neighbors Classifier", KNeighborsClassifier.predict(data)])
    result_set.append(["LogisticRegression", LogisticRegression.predict(data)])
    result_set.append(["naiveBayes", naiveBayes.predict(data)])
    result_set.append(["RandomForestClassifier", RandomForestClassifier.predict(data)])
    result_set.append(["SVM", SVM.predict(data)])
    
    return result_set

    
    
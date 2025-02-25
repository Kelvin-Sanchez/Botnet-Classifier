"""
Kelvin Sanchez
IoT Botnet Detection

Creates a Random Forest Classifier and a Multi-layer Perceptron Classifier
Models are trained on the Botnet_train.csv
Models are tested on the Botnet_test.csv
Pickle used to export models for use in predict.py
"""


from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle


def get_dataset(in_file):
    data = pd.read_csv(in_file)
    label = data['attack']
    data = data.drop(["pkSeqID", "saddr", "daddr", "sport", "dport","attack", "proto", "category", "subcategory"], axis=1)
    features = data
    return label, features


y_train, x_train = get_dataset('botnet_train.csv')
y_test, x_test = get_dataset('botnet_test.csv')

pca = PCA(n_components=7, whiten=True)
pca.fit(x_train)
pca_train = pca.transform(x_train)
pca_test = pca.transform(x_test)

rfc = RandomForestClassifier(criterion='log_loss', max_features='sqrt', min_samples_split= 2, n_estimators= 100, n_jobs=-1)

rfc.fit(pca_train, y_train)
rfc_pred = rfc.predict(pca_test)
print("RFC Scores")
print(classification_report(y_test, rfc_pred))
print(accuracy_score(y_test, rfc_pred))

mlp = MLPClassifier(activation='logistic', learning_rate='invscaling', max_iter=400, solver='adam')

mlp.fit(x_train, y_train)
mlp_pred = mlp.predict(x_test)
print("MLP Scores")
print(classification_report(y_test, mlp_pred))
print(accuracy_score(y_test, mlp_pred))

pickle.dump((rfc, pca), open("RFC.mod",'wb'))
pickle.dump(mlp, open("MLP.mod",'wb'))
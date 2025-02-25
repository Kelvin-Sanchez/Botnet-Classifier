import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def get_dataset(in_file):
    data = pd.read_csv(in_file)
    label = data['attack']
    data = data.drop(["pkSeqID", "saddr", "daddr", "sport", "dport","attack", "proto", "category", "subcategory"],
                     axis=1)
    features = data
    return label, features


rfc, pca = pickle.load(open("RFC.mod", "rb"))
mlp = pickle.load(open("MLP.mod", "rb"))

file = input("Please enter the file name for the sample: ")

y_test, x_test = get_dataset(file)

pca_test = pca.transform(x_test)
rfc_pred = rfc.predict(pca_test)

for pred in range(len(rfc_pred)):
    print("Line: " + str(pred) + " Classified as " + str(rfc_pred[pred]))

print("RFC Scores")
print(classification_report(y_test, rfc_pred))
print("Accuracy Score: ",accuracy_score(y_test, rfc_pred))
print("*****************************************************")
mlp_pred = mlp.predict(x_test)

for pred in range(len(mlp_pred)):
    print("Line: " + str(pred) + " Classified as " + str(mlp_pred[pred]))

print("MLP Scores")
print(classification_report(y_test, mlp_pred))
print("Accuracy Score: ",accuracy_score(y_test, mlp_pred))
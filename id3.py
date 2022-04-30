from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import pickle

# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = np.nan)
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(np.ravel(y_train))
    y_train_enc = le.transform(np.ravel(y_train))
    y_test_enc = le.transform(np.ravel(y_test))
    return y_train_enc, y_test_enc


def load_dataset(filename):

    data = read_csv(filename)
    dataset = data.values
    X = dataset[:, :-1]
    y = dataset[:, -1]
    X = X.astype(str)
    y = y.reshape((len(y), 1))
    return X, y


X, y = load_dataset('heart_disease_male(csv).csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

print(X_test)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
oe.fit(X_train)

# Encode data
X_train_enc = oe.transform(X_train)
X_test_enc = oe.transform(X_test)

y_train_enc , y_test_enc = prepare_targets(y_train, y_test)

X_train_enc = np.nan_to_num(X_train_enc)
X_test_enc = np.nan_to_num(X_test_enc)
y_train_enc = np.nan_to_num(y_train_enc)
y_test_enc = np.nan_to_num(y_test_enc)

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X_train_enc, y_train_enc)

result = decision_tree.predict(X_test_enc)

print(result)

pickle.dump(decision_tree, open('id3_model.pkl','wb'))

id3_model = pickle.load(open('id3_model.pkl', 'rb'))

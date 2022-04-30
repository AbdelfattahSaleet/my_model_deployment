from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
oe.fit(X_train)
X_train_enc = oe.transform(X_train)
X_test_enc = oe.transform(X_test)

y_train_enc , y_test_enc = prepare_targets(y_train, y_test)

X_train_enc = np.nan_to_num(X_train_enc)
X_test_enc = np.nan_to_num(X_test_enc)
y_train_enc = np.nan_to_num(y_train_enc)
y_test_enc = np.nan_to_num(y_test_enc)

bnb = BernoulliNB()

p = bnb.fit(X_train_enc, y_train_enc).predict(X_test_enc)

acc = accuracy_score(y_test_enc, y_pred)
MCn = (y_test_enc != y_pred).sum()
mcr = MCn / y_test_enc.shape[0]

print("Number of mislabeled points out of a total %d points: %d" %(X_test_enc.shape[0], MCn))
print('accuracy: ', acc * 100, ',')
print('misclassification Rate ', mcr * 100, ',')

pickle.dump(bnb, open('naive_bayes_model.pkl','wb'))

model = pickle.load(open('naive_bayes_model.pkl', 'rb'))

import pandas, numpy
import scipy.io
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier


# import training dataset
texture = pandas.read_csv('training 1featureMean 3classes TTP-TTP+120s withtime offset2 0312.csv')
array = texture.values
# shuffle dataset
numpy.random.shuffle(array)
# select predictor variables
x = array[:, 0]
x = x.reshape(-1, 1)
# select response variables
y = array[:, 3]

# load testing dataset from .mat
mat = scipy.io.loadmat('DHA007 icg02 TTP+120s test-FI.mat')
print(sorted(mat.keys()))
X_test = mat['Texture12']
X_test = X_test.reshape(-1, 1)
# logistic regression classifier
log_class = LogisticRegression(solver='liblinear', C=1, random_state=0, penalty='l2')
ovo_log = OneVsOneClassifier(log_class)
ovo_log.fit(x, y)
y_pred = ovo_log.predict(X_test)
# make and save resulting table
resulttable = pandas.DataFrame(data=y_pred)
resulttable.to_csv (r'C:\Users\f00349n\Desktop\Python Scripts\DHA007 icg02 TTP+120s FI-ypredict.csv', header=False)
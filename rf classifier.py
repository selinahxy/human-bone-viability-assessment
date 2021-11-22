import pandas, numpy
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.decomposition import PCA


# import training dataset
texture = pandas.read_csv('training footoff 20features 3classes 7patients round2 by DecisionOutlines TTP-TTP+120s withtime offset2nl256 1014.csv')
array = texture.values
# shuffle dataset
numpy.random.shuffle(array)
# select predictor variables
x = array[:, 0:20]
# select response variables
y = array[:, 23]
# principle component analysis
pca = PCA(n_components=3)
pca.fit(x)
pc_train = pca.transform(x)

# load testing dataset
texture_test = pandas.read_csv('testing A01-icg01 all pixels 20features 3classes by 7patients round2 3d Clustering weighted features TTP+110s offset2nl256 1014.csv')
texture_test = texture_test.fillna(0)
array_test = texture_test.values
# shuffle dataset
numpy.random.shuffle(array_test)
# select predictor variables
X_test = array_test[:, 0:20]
pc_test = pca.transform(X_test)
# logistic regression classifier
log_class = LogisticRegression(solver='liblinear', C=1, random_state=0, penalty='l2')
ovo_log = OneVsOneClassifier(log_class)
ovo_log.fit(pc_train, y)
y_pred = ovo_log.predict(pc_test)
# make and save resulting table
resulttable = pandas.DataFrame(data=y_pred)
resulttable.to_csv (r'C:\Users\f00349n\Desktop\Python Scripts\DHA001 icg01 TTP+110s LG-ypredict.csv', header=False)

import pandas, numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.decomposition import PCA


# import organized training dataset
texture = pandas.read_csv('training footoff 20features 3classes 7patients round2 by DecisionOutlines TTP-TTP+120s withtime offset2nl256 1014.csv')
array = texture.values
# shuffle dataset
numpy.random.shuffle(array)
# select predictor variables
x = array[:, 0:20]
# rescale each column to [0,1]
scaler = MinMaxScaler().fit(x)
rescaledX = scaler.transform(x)
# select response variables
y = array[:, 23]
# select grouping variables
groups_patient = array[:, 21]
gkf = GroupKFold(n_splits=7)

log_validation_acc = []
log_test_acc = []
log_FNR = []
log_FPR = []
log_SB = []
rf_validation_acc = []
rf_test_acc = []
rf_FNR = []
rf_FPR = []
rf_SB = []
gb_validation_acc = []
gb_test_acc = []
gb_FNR = []
gb_FPR = []
gb_SB = []
knn_validation_acc = []
knn_test_acc = []
knn_FNR = []
knn_FPR = []
knn_SB = []


# grouped 7-fold cross validation
for train, test in gkf.split(rescaledX, y, groups=groups_patient):
    X_train, X_test, y_train, y_test = rescaledX[train], rescaledX[test], y[train], y[test]
    # principle component analysis
    pca = PCA(n_components=3)
    pca.fit(X_train)
    pc_train = pca.transform(X_train)
    pc_test = pca.transform(X_test)
    # logistic regression classifier
    log_class = LogisticRegression(solver='liblinear', C=1, random_state=0, penalty='l2')
    ovo_log = OneVsOneClassifier(log_class)  # one vs. one strategy
    ovo_log.fit(pc_train, y_train)
    y_pred = ovo_log.predict(pc_test)
    log_validation_acc.append(ovo_log.score(pc_train, y_train))
    log_test_acc.append(ovo_log.score(pc_test, y_test))
    cm1 = confusion_matrix(y_test, y_pred)
    FNR1 = (cm1[1, 0] + cm1[2, 0])/(cm1[1, 0] + cm1[1, 1] + cm1[1, 2] + cm1[2, 0] + cm1[2, 1] + cm1[2, 2])  # cost function FNR
    FPR1 = (cm1[0, 1] + cm1[0, 2])/(cm1[0, 0] + cm1[0, 1] + cm1[0, 2])  # cost function FPR
    SB1 = (cm1[0, 1] + cm1[1, 1] + cm1[2, 1])/(cm1[1, 0] + cm1[1, 1] + cm1[1, 2])  # cost function SB
    log_FNR.append(FNR1)
    log_FPR.append(FPR1)
    log_SB.append(SB1)
    # random forest classifier
    rf_class = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_class.fit(pc_train, y_train)
    y_pred = rf_class.predict(pc_test)
    rf_validation_acc.append(rf_class.score(pc_train, y_train))
    rf_test_acc.append(rf_class.score(pc_test, y_test))
    cm3 = confusion_matrix(y_test, y_pred)
    FNR3 = (cm3[1, 0] + cm3[2, 0]) / (cm3[1, 0] + cm3[1, 1] + cm3[1, 2] + cm3[2, 0] + cm3[2, 1] + cm3[2, 2])
    FPR3 = (cm3[0, 1] + cm3[0, 2])/(cm3[0, 0] + cm3[0, 1] + cm3[0, 2])
    SB3 = (cm3[0, 1] + cm3[1, 1] + cm3[2, 1])/(cm3[1, 0] + cm3[1, 1] + cm3[1, 2])
    rf_FNR.append(FNR3)
    rf_FPR.append(FPR3)
    rf_SB.append(SB3)
    # gradient boosting
    gb_class = GradientBoostingClassifier(n_estimators=100, random_state=0)
    ovo_gb = OneVsOneClassifier(gb_class)
    ovo_gb.fit(pc_train, y_train)
    y_pred = ovo_gb.predict(pc_test)
    gb_validation_acc.append(ovo_gb.score(pc_train, y_train))
    gb_test_acc.append(ovo_gb.score(pc_test, y_test))
    cm4 = confusion_matrix(y_test, y_pred)
    FNR4 = (cm4[1, 0] + cm4[2, 0]) / (cm4[1, 0] + cm4[1, 1] + cm4[1, 2] + cm4[2, 0] + cm4[2, 1] + cm4[2, 2])
    FPR4 = (cm4[0, 1] + cm4[0, 2])/(cm4[0, 0] + cm4[0, 1] + cm4[0, 2])
    SB4 = (cm4[0, 1] + cm4[1, 1] + cm4[2, 1])/(cm4[1, 0] + cm4[1, 1] + cm4[1, 2])
    gb_FNR.append(FNR4)
    gb_FPR.append(FPR4)
    gb_SB.append(SB4)
    # knn classifier
    knn_class = KNeighborsClassifier(n_neighbors=5)
    ovo_knn = OneVsOneClassifier(knn_class)
    ovo_knn.fit(pc_train, y_train)
    y_pred = ovo_knn.predict(pc_test)
    knn_validation_acc.append(ovo_knn.score(pc_train, y_train))
    knn_test_acc.append(ovo_knn.score(pc_test, y_test))
    cm5 = confusion_matrix(y_test, y_pred)
    FNR5 = (cm5[1, 0] + cm5[2, 0]) / (cm5[1, 0] + cm5[1, 1] + cm5[1, 2] + cm5[2, 0] + cm5[2, 1] + cm5[2, 2])
    FPR5 = (cm5[0, 1] + cm5[0, 2])/(cm5[0, 0] + cm5[0, 1] + cm5[0, 2])
    SB5 = (cm5[0, 1] + cm5[1, 1] + cm5[2, 1])/(cm5[1, 0] + cm5[1, 1] + cm5[1, 2])
    knn_FNR.append(FNR5)
    knn_FPR.append(FPR5)
    knn_SB.append(SB5)


# make and save resulting table
result = numpy.array([[numpy.mean(log_validation_acc), numpy.mean(log_test_acc), numpy.mean(log_FNR), numpy.nanmean(log_FPR), numpy.nanmean(log_SB),
                       numpy.std(log_test_acc), numpy.std(log_FNR), numpy.nanstd(log_FPR), numpy.nanmean(log_SB)],
                      [numpy.mean(rf_validation_acc), numpy.mean(rf_test_acc), numpy.mean(rf_FNR), numpy.nanmean(rf_FPR), numpy.nanmean(rf_SB),
                       numpy.std(rf_test_acc), numpy.std(rf_FNR), numpy.nanstd(rf_FPR), numpy.nanmean(rf_SB)],
                      [numpy.mean(gb_validation_acc), numpy.mean(gb_test_acc), numpy.mean(gb_FNR), numpy.nanmean(gb_FPR), numpy.nanmean(gb_SB),
                       numpy.std(gb_test_acc), numpy.std(gb_FNR), numpy.nanstd(gb_FPR), numpy.nanmean(gb_SB)],
                      [numpy.mean(knn_validation_acc), numpy.mean(knn_test_acc), numpy.mean(knn_FNR), numpy.nanmean(knn_FPR), numpy.nanmean(knn_SB),
                       numpy.std(knn_test_acc), numpy.std(knn_FNR), numpy.nanstd(knn_FPR), numpy.nanmean(knn_SB)]])
resulttable = pandas.DataFrame(data=result, index=["log", "random forest", "gradient boosting", "knn"],
                               columns=["train accuracy", "validation accuracy", "FNR", "FPR", "SB",
                                        "std:validation accuracy", "std:sensitivity_semi", "std:sensitivity_non", "std:SB"])
resulttable.to_csv (r'C:\Users\f00349n\Desktop\Python Scripts\20F3C fTTP-TTP120s decisionoutline 7patient footoff round2 pca nl256 corrected.csv', header=True)
print(resulttable)

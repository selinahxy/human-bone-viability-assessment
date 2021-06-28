import pandas, numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeavePGroupsOut
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.decomposition import PCA


# import organized training dataset
texture = pandas.read_csv('training 20features 3classes by 3d Clustering TTP-TTP+120s withtime offset2nl256 0328.csv')
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
y = numpy.where(y == 2, 0, y)
y = numpy.where(y == 3, 2, y)
# select grouping variables
groups_patient = array[:, 21]
gkf = GroupKFold(n_splits=3)

log_validation_acc = []
log_test_acc = []
log_sensitivity_de = []
log_sensitivity_no = []
svm_validation_acc = []
svm_test_acc = []
svm_sensitivity_de = []
svm_sensitivity_no = []
rf_validation_acc = []
rf_test_acc = []
rf_sensitivity_de = []
rf_sensitivity_no = []
gb_validation_acc = []
gb_test_acc = []
gb_sensitivity_de = []
gb_sensitivity_no = []
knn_validation_acc = []
knn_test_acc = []
knn_sensitivity_de = []
knn_sensitivity_no = []


# grouped 3-fold cross validation
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
    if numpy.array(cm1).size == 4:
        cm11 = numpy.array([[cm1[0, 0], cm1[0, 1], 0],
                        [cm1[1, 0], cm1[1, 1], 0],
                        [0, 1, 0]])
    else:
        cm11 = cm1
    sensitivity1 = cm11[1, 1] / (cm11[1, 0] + cm11[1, 1] + cm11[1, 2])
    sensitivity12 = cm11[2, 2] / (cm11[2, 0] + cm11[2, 1] + cm11[2, 2])
    log_sensitivity_de.append(sensitivity1)
    log_sensitivity_no.append(sensitivity12)
    # svm
    svm_class = svm.SVC(kernel='rbf')
    ovo_svm = OneVsOneClassifier(svm_class)
    ovo_svm.fit(pc_train, y_train)
    y_pred = ovo_svm.predict(pc_test)
    svm_validation_acc.append(ovo_svm.score(pc_train, y_train))
    svm_test_acc.append(ovo_svm.score(pc_test, y_test))
    cm2 = confusion_matrix(y_test, y_pred)
    if numpy.array(cm2).size == 4:
        cm22 = numpy.array([[cm2[0, 0], cm2[0, 1], 0],
                        [cm2[1, 0], cm2[1, 1], 0],
                        [0, 1, 0]])
    else:
        cm22 = cm2
    sensitivity2 = cm22[1, 1] / (cm22[1, 0] + cm22[1, 1] + cm22[1, 2])
    sensitivity22 = cm22[2, 2] / (cm22[2, 0] + cm22[2, 1] + cm22[2, 2])
    svm_sensitivity_de.append(sensitivity2)
    svm_sensitivity_no.append(sensitivity22)
    # random forest classifier
    rf_class = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_class.fit(pc_train, y_train)
    y_pred = rf_class.predict(pc_test)
    rf_validation_acc.append(rf_class.score(pc_train, y_train))
    rf_test_acc.append(rf_class.score(pc_test, y_test))
    cm3 = confusion_matrix(y_test, y_pred)
    if numpy.array(cm3).size == 4:
        cm33 = numpy.array([[cm3[0, 0], cm3[0, 1], 0],
                            [cm3[1, 0], cm3[1, 1], 0],
                            [0, 1, 0]])
    else:
        cm33 = cm3
    sensitivity3 = cm33[1, 1] / (cm33[1, 0] + cm33[1, 1] + cm33[1, 2])
    sensitivity32 = cm33[2, 2] / (cm33[2, 0] + cm33[2, 1] + cm33[2, 2])
    rf_sensitivity_de.append(sensitivity3)
    rf_sensitivity_no.append(sensitivity32)
    # gradient boosting
    gb_class = GradientBoostingClassifier(n_estimators=100, random_state=0)
    ovo_gb = OneVsOneClassifier(gb_class)
    ovo_gb.fit(pc_train, y_train)
    y_pred = ovo_gb.predict(pc_test)
    gb_validation_acc.append(ovo_gb.score(pc_train, y_train))
    gb_test_acc.append(ovo_gb.score(pc_test, y_test))
    cm4 = confusion_matrix(y_test, y_pred)
    if numpy.array(cm4).size == 4:
        cm43 = numpy.array([[cm4[0, 0], cm4[0, 1], 0],
                            [cm4[1, 0], cm4[1, 1], 0],
                            [0, 1, 0]])
    else:
        cm43 = cm4
    sensitivity4 = cm43[1, 1] / (cm43[1, 0] + cm43[1, 1] + cm43[1, 2])
    sensitivity42 = cm43[2, 2] / (cm43[2, 0] + cm43[2, 1] + cm43[2, 2])
    gb_sensitivity_de.append(sensitivity4)
    gb_sensitivity_no.append(sensitivity42)
    # knn classifier
    knn_class = KNeighborsClassifier(n_neighbors=5)
    ovo_knn = OneVsOneClassifier(knn_class)
    ovo_knn.fit(pc_train, y_train)
    y_pred = ovo_knn.predict(pc_test)
    knn_validation_acc.append(ovo_knn.score(pc_train, y_train))
    knn_test_acc.append(ovo_knn.score(pc_test, y_test))
    cm5 = confusion_matrix(y_test, y_pred)
    if numpy.array(cm5).size == 4:
        cm55 = numpy.array([[cm5[0, 0], cm5[0, 1], 0],
                        [cm5[1, 0], cm5[1, 1], 0],
                        [0, 1, 0]])
    else:
     cm55 = cm5
    sensitivity5 = cm55[1, 1] / (cm55[1, 0] + cm55[1, 1] + cm55[1, 2])
    sensitivity52 = cm55[2, 2] / (cm55[2, 0] + cm55[2, 1] + cm55[2, 2])
    knn_sensitivity_de.append(sensitivity5)
    knn_sensitivity_no.append(sensitivity52)


# make and save resulting table
result = numpy.array([[numpy.mean(log_validation_acc), numpy.mean(log_test_acc), numpy.mean(log_sensitivity_de), numpy.nanmean(log_sensitivity_no),
                       numpy.std(log_test_acc), numpy.std(log_sensitivity_de), numpy.nanstd(log_sensitivity_no)],
                      [numpy.mean(svm_validation_acc), numpy.mean(svm_test_acc), numpy.mean(svm_sensitivity_de), numpy.nanmean(svm_sensitivity_no),
                       numpy.std(svm_test_acc), numpy.std(svm_sensitivity_de), numpy.nanstd(svm_sensitivity_no)],
                      [numpy.mean(rf_validation_acc), numpy.mean(rf_test_acc), numpy.mean(rf_sensitivity_de), numpy.nanmean(rf_sensitivity_no),
                       numpy.std(rf_test_acc), numpy.std(rf_sensitivity_de), numpy.nanstd(rf_sensitivity_no)],
                      [numpy.mean(gb_validation_acc), numpy.mean(gb_test_acc), numpy.mean(gb_sensitivity_de), numpy.nanmean(gb_sensitivity_no),
                       numpy.std(gb_test_acc), numpy.std(gb_sensitivity_de), numpy.nanstd(gb_sensitivity_no)],
                      [numpy.mean(knn_validation_acc), numpy.mean(knn_test_acc), numpy.mean(knn_sensitivity_de), numpy.nanmean(knn_sensitivity_no),
                       numpy.std(knn_test_acc), numpy.std(knn_sensitivity_de), numpy.nanstd(knn_sensitivity_no)]])
resulttable = pandas.DataFrame(data=result, index=["log", "SVM", "random forest", "gradient boosting", "knn"],
                               columns=["train accuracy", "validation accuracy", "sensitivity_semi", "sensitivity_non",
                                        "std:validation accuracy", "std:sensitivity_semi", "std:sensitivity_non"])
resulttable.to_csv (r'C:\Users\f00349n\Desktop\Python Scripts\20F3C fTTP-TTP120s 3dclustering pca nl256 2sensitivities add std.csv', header=True)
print(resulttable)

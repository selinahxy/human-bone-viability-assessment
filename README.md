# human-bone-viability-assessment
This is the first human bone viability assessment approach using texture analysis of DCE-FI (Dynamic Contrast Enhanced- Fluorescence Imaging), by unsupervised learning algorithm with K-means clustering and PCA.

training.m is the MATLAB code of creating training dataset and train the algorithm. It contents processing raw DICOM fluorescence videos, image texture feature extraction, organizing into data table, PCA and K-means clustering, and FI (Fluorescence Intensities) extraction from manually labelled ROIs.

GLCM_Features.m is adapted from ref[1]. It computes a subset of gray co-occurence matrix features in a verctorized fashion, and outputs 13 Haralick's features.

testing.m is the MATLAB code of generating testing dataset and test the algorithm. It contents processing testing DICOM fluorescence videos, image texture feature extraction, PCA and applying to training K-means clustering, comparing with ground-truth decision outline maps, and testing FI classifier.

imageFeature.m is a modified version of GLCM_Features.m. It can go through a region of interest selected by binary mask, and compute image texture features pixel-by-pixel. This function is used in testing dataset to produce high resolution resulting maps.

training_FI_and_supervised_learning.m is the MATLAB code of creating training dataset for FI threshold classifier and supervised learning classifier. It contents input ROI labels, save dataset, and FI thresholding exhausted search.

testing_FI_and_supervised_learning.m is the MATLAB code of testing the FI and supervised learning. It contents extract FI from testing patients, compute cost functions by FI and supervised learning.

texture classifier 3class pca.py is the Python code of testing the K-means clustering classifier by grouped k-fold cross validation. The machine learning algorithms included are logistic regression, random forest, gradient boosting and KNN. It uses the first three principle components as predictors, clustering numbers as response variables, and patient ID as group index. After k iterations (k=3 here), it computes the average total accuracy, false-negative-rate (FNR), false-positive-rate (FPR) and surgeon's burden (SB), as well as the standard deviations.

rf classifier.py is the Python code of testing the FI classifier. It loads the testing dataset in .mat, and produces the computed response class according to logistic regression classifier by input FI predictors.

feature heatmap.py is the Python code of generating heatmap for texture features. It uses the seaborn.clustermap package.

Ref:
[1] Pawel Kleczek (2021). GLCM_Features(glcm) (https://www.mathworks.com/matlabcentral/fileexchange/56661-glcm_features-glcm), MATLAB Central File Exchange. Retrieved June 28, 2021.
[2] Avinash Uppuluri (2021). GLCM texture features (https://www.mathworks.com/matlabcentral/fileexchange/22187-glcm-texture-features), MATLAB Central File Exchange. Retrieved June 28, 2021.
[3] Haralick RM, Shanmuga K, Dinstein I.: Textural features for image classification. IEEE Trans Syst Man Cybern 3: 610-621, 1973

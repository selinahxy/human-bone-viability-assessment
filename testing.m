%% feature extraction of testing dataset
close all

% Constants
PRE_TIME = 10; %s - time before ICG injection (conservative estimate).

% open DICOM stack or MAT
[filename, pathname] = uigetfile('*.*', 'Pick the SPY Elite dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA-009' );
%     pathname = 'X:\#5 - Data\Human Bone Fracture DCE-FI\001_BoneFI_20190516_083500\01ICG_20190516_083500\';
%     filename = '01ICG_20190516_SPY_FI';

[rgb_file,~]=uigetfile('*.*', 'Pick the White light dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA-009' );

% DICOM
info = dicominfo( [pathname filename] );
d.t = linspace(0, (info.NumberOfFrames-1)*(info.FrameTime/1000), info.NumberOfFrames);
dicom_data = dicomread( [pathname filename] );

d.FI = squeeze(double(dicom_data));
d.Q = d.FI - mean( d.FI(:,:,1:find(d.t > PRE_TIME,1,'first')), 3 );


[X,Y,Z] = size(d.Q);
Q_l = reshape(d.Q, [X*Y Z]);
[Q_max_l, I_max_l] = max(Q_l,[],2);
d.Q_max = reshape(Q_max_l,[X Y]);
d.TTP = reshape(d.t(I_max_l),[X Y]);

% check for RGB images
if exist('rgb_file')
    d.RGB = squeeze(dicomread([pathname,rgb_file]));
end

% line to select the orientation of the image (rotate), should be parallel
% % to what you want the x-axis to be.
imagesc(d.RGB); 
colormap(goodmap('linearL'));axis off;
h2 = imline;
pos2 = getPosition(h2);
theta=atan((pos2(1,2)-pos2(2,2))/(pos2(1,1)-pos2(1,2)));

% rotate image.
d.c.RGB = imrotate(d.RGB,rad2deg(theta),'bilinear');
d.c.TTP = imrotate(d.TTP,rad2deg(theta),'bilinear');
d.c.Q = zeros(size(d.c.RGB,1),size(d.c.RGB,2),Z);
for i = 1:Z
    d.c.Q(:,:,i) = imrotate(d.Q(:,:,i),rad2deg(theta),'bilinear');
end

% select rectanglar region of interest
imagesc(d.c.RGB);
h0 = drawrectangle('StripeColor','k');
% select rectanglar region of background
h5 = drawrectangle;
bw=createMask(h0);

frame=zeros(1,13);

%save texture analysis results for every 10 seconds starting from
%Time-To-Peak to 120s after
for i=1:13
    frame(i)=round(130+37.5*(i-1));
    if frame(i)<=1024
        m=frame(i);
        I=d.c.Q(:,:,m);
        Back=imcrop(I,h5.Position);
        Background(i)=nanmean(Back(:));
        ROI1=imcrop(I,h0.Position);
        % feature extraction on pixel-by-pixel sliding sub-ROIs
        [psta_256]=imageFeature(I,bw,256,2,21,1);
        psta_256(:,:,20)=(psta_256(:,:,14)-Background(i))/nanstd(Back(:));
        save(['DHA001 0916 icg03 texture o2f21nl256sub20 TTP+',num2str(i-1),'0s.mat'],'psta_256');
        clear psta_256
        disp(i);
    end
end

%% PCA and K-means clustering on testing patients(add testing data on exsiting PCA and clustering)
% import organized training dataset
data1=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P16 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data2=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P12 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data3=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P13 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data4=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P10 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data5=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A08 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data6=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A09 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data7=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A10 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');

% import testing dataset from each testing patients individually
test0=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+00s.mat');
test1=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+10s.mat');
test2=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+20s.mat');
test3=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+30s.mat');
test4=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+40s.mat');
test5=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+50s.mat');
test6=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+60s.mat');
test7=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+70s.mat');
test8=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+80s.mat');
test9=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+90s.mat');
test10=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+100s.mat');
test11=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+110s.mat');
test12=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New testing\DHA07 0916 icg01 texture o2f21nl256sub20 TTP+120s.mat');


[xx0,yy0,zz] = size(test0);
[xx1,yy1,zz] = size(test1);
[xx2,yy2,zz] = size(test2);
[xx3,yy3,zz] = size(test3);
[xx4,yy4,zz] = size(test4);
[xx5,yy5,zz] = size(test5);
[xx6,yy6,zz] = size(test6);
[xx7,yy7,zz] = size(test7);
[xx8,yy8,zz] = size(test8);
[xx9,yy9,zz] = size(test9);
[xx10,yy10,zz] = size(test10);
[xx11,yy11,zz] = size(test11);
[xx12,yy12,zz] = size(test12);
XTest0=[];
XTest1=[];
XTest2=[];
XTest3=[];
XTest4=[];
XTest5=[];
XTest6=[];
XTest7=[];
XTest8=[];
XTest9=[];
XTest10=[];
XTest11=[];
XTest12=[];

% organize testing dataset
for i=1:zz
    XTest0=[XTest0 reshape(test0(:,:,i),[xx0*yy0 1])];
    XTest1=[XTest1 reshape(test1(:,:,i),[xx1*yy1 1])];
    XTest2=[XTest2 reshape(test2(:,:,i),[xx2*yy2 1])];
    XTest3=[XTest3 reshape(test3(:,:,i),[xx3*yy3 1])];
    XTest4=[XTest4 reshape(test4(:,:,i),[xx4*yy4 1])];
    XTest5=[XTest5 reshape(test5(:,:,i),[xx5*yy5 1])];
    XTest6=[XTest6 reshape(test6(:,:,i),[xx6*yy6 1])];
    XTest7=[XTest7 reshape(test7(:,:,i),[xx7*yy7 1])];
    XTest8=[XTest8 reshape(test8(:,:,i),[xx8*yy8 1])];
    XTest9=[XTest9 reshape(test9(:,:,i),[xx9*yy9 1])];
    XTest10=[XTest10 reshape(test10(:,:,i),[xx10*yy10 1])];
    XTest11=[XTest11 reshape(test11(:,:,i),[xx11*yy11 1])];
    XTest12=[XTest12 reshape(test12(:,:,i),[xx12*yy12 1])];
end
XTest=[XTest0;XTest1;XTest2;XTest3;XTest4;XTest5;XTest6;XTest7;XTest8;XTest9;XTest10;XTest11;XTest12];
XTest_id=[zeros(xx0*yy0,1);ones(xx1*yy1,1);ones(xx2*yy2,1).*2;ones(xx3*yy3,1).*3;ones(xx4*yy4,1).*4;ones(xx5*yy5,1).*5;ones(xx6*yy6,1).*6;...
    ones(xx7*yy7,1).*7;ones(xx9*yy9,1).*9;ones(xx10*yy10,1).*10;ones(xx11*yy11,1).*11;ones(xx12*yy12,1).*12];

%normalize the testing patient data to [0,1]
XPredictors=normalize(XTest,'range');

%normalize the training patient data to [0,1]
Texture=[normalize(data1(:,1:20),'range');normalize(data2(:,1:20),'range');normalize(data3(:,1:20),'range');...
    normalize(data4(:,1:20),'range');normalize(data5(:,1:20),'range');normalize(data6(:,1:20),'range');normalize(data7(:,1:20),'range')];

% principle component analysis
[coeff,score,~,~,explained,mu]=pca(Texture);
Xcentered=score*coeff';

% k-means clustering (k=3) on the top three PCs weighted by square root of
% their percentages of total variance explained
X=[score(:,1).*realsqrt(round(explained(1)/10)),score(:,2).*realsqrt(round(explained(2)/10)),...
    score(:,3).*realsqrt(round(explained(3)/10))];
opts = statset('Display','final');
[idx,C] = kmeans(X,3,'Replicates',5,'Options',opts);

% apply training PCA to testing data
scoreTest95_before = (XPredictors-mu)*coeff(:,1:3); 
scoreTest95=[scoreTest95_before(:,1).*realsqrt(round(explained(1)/10)),scoreTest95_before(:,2).*realsqrt(round(explained(2)/10)),...
    scoreTest95_before(:,3).*realsqrt(round(explained(3)/10))];
% assign testing data to existing clusters by their nearest centroid
[~,idx_test] = pdist2(C,scoreTest95,'euclidean','Smallest',1); 

% 3D scatter plot of testing data partitioned by k-means clustering
figure(2)
scatter3(X(idx==1,1),X(idx==1,2),X(idx==1,3),36,'y.')
hold on
scatter3(X(idx==2,1),X(idx==2,2),X(idx==2,3),36,'g.')
scatter3(X(idx==3,1),X(idx==3,2),X(idx==3,3),36,'r.')
scatter3(scoreTest95(idx_test==1,1),scoreTest95(idx_test==1,2),scoreTest95(idx_test==1,3),'MarkerEdgeColor','k','MarkerFaceColor','y')
scatter3(scoreTest95(idx_test==2,1),scoreTest95(idx_test==2,2),scoreTest95(idx_test==2,3),'MarkerEdgeColor','k','MarkerFaceColor','g')
scatter3(scoreTest95(idx_test==3,1),scoreTest95(idx_test==3,2),scoreTest95(idx_test==3,3),'MarkerEdgeColor','k','MarkerFaceColor','r') 
legend('Cluster 1','Cluster 2','Cluster 3',...
        'Location','NW')
xlabel('Component 1')
ylabel('Component 2')
zlabel('Component 3')
hold off

% save the data table and add the column of clustering number
Cleaned=[XTest,idx_test'];
for i=0:0
Cleaned_n=Cleaned(XTest_id==i,:);
Cleanedtable=array2table(Cleaned_n,'VariableNames',{'256-GC-Contrast','256-Energy','256-SumOfSquares','256-Entropy','256-Homogeneity','256-SumAverage',...
    '256-SumEntropy','256-SumVairance','256-DifferenceEntropy','256-DifferenceVariance','256-IMFCorrelation1','256-IMFCorrelation2','256-Correlation',...
    'Mean','STD','Skewness','Kurtosis','IM-Contrast','gamma-k','gamma-phi','Class'});
writetable(Cleanedtable,['testing A07-icg01 all pixels 20features 3classes by 7patients round2 3d Clustering weighted features TTP+',num2str(i),'0s offset2nl256 1014.csv']);
clear Cleaned_n Cleanedtable
end

 %% overlay with decision outline images and calculate confusion matrix: k-means clustering classifier
% load outline image
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007 POST BONE CUT AND POST SOFT TISSUE STRIPPING\exportedStudy-20210422-00030\icg02-RGB outlines.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\AMPUTATION\DHA-001 SPY\exportedStudy-20200805-00351\exportedSequence-004_output\exportedSequence-004_sample.jpg');
I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007PRE BONE CUT\exportedStudy-20210422-00029\icg01-outline.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA001\icg01\icg01-outline.jpg');

% resize the outline image to be the same as k-means clustering map 
I_resize=imresize(I,[748 1004]);                  

imshow(I_resize);
axis off;
h0 = drawfreehand('closed');
bw1=createMask(h0);

% iterate through every 10 seconds starting from Time-To-Peak to 120s after
for i=[0:7,9:11]
    % k-means clustering predictive map
    fs_table=readtable(['C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\testing A07-icg01 all pixels 20features 3classes by 7patients round2 3d Clustering weighted features TTP+',num2str(i),'0s offset2nl256 1014.csv']);
    fs=table2array(fs_table);
    selectedn=fs(:,21);
    label_image = reshape(selectedn, [748,1004]);
    filled_ROI1=imfill(label_image,8); %hole-filling processing
    filled_image=filled_ROI1.*(double(bw1));
   
    figure(4*i+2)
    % overlay outline image with k-means clustering map
    ii=imshow(I_resize);
    hold on;
    im=imagesc(filled_image);
    colormap([0 1 0;1 1 0;1 0 0]);
    im.AlphaData = 0.4;
    colorbar;
    axis off;
    hold off;
   
    ypredict_cluster=nonzeros(filled_image);
    yactual_cluster=ones(size(ypredict_cluster,1),1).*3;
    
    figure(4*i+4)
    % confusion matrix of k-means clustering
    C1=confusionmat(yactual_cluster,ypredict_cluster);
    confusionchart(C1)

end

%% decision outlines map: Figure 5(a) Column 2
% load outline image
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007 POST BONE CUT AND POST SOFT TISSUE STRIPPING\exportedStudy-20210422-00030\icg02-RGB outlines.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\AMPUTATION\DHA-001 SPY\exportedStudy-20200805-00351\exportedSequence-004_output\exportedSequence-004_sample.jpg');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007PRE BONE CUT\exportedStudy-20210422-00029\icg01_RGB_outlines.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA001\icg01\icg01-RGB_outlines.png');
I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\016_BoneFI_20200227\P16-icg03-RGB--outlines.png');

I_resize=imresize(I,[768 1024]);

imshow(I_resize);
axis off;
h0 = drawfreehand('closed');
bw0=createMask(h0);
h = drawfreehand('closed');
bw1=createMask(h);
h2 = drawfreehand('closed');
bw2=createMask(h2);
I0=ones(768,1024).*double(bw0);
I1=2.*ones(768,1024).*double(bw1);
I2=ones(768,1024).*double(bw2);
I_t=I0+I1+I2;

figure(2)
ii=imshow(I_resize);
hold on;
im=imagesc(I_t);
colormap([1 1 1;1 1 0;1 0 0]);
im.AlphaData = 0.4;
colorbar;
hold off;

%% feature extraction of testing dataset
close all

% Constants
PRE_TIME = 10; %s - time before ICG injection (conservative estimate).

% open DICOM stack or MAT
[filename, pathname] = uigetfile('*.*', 'Pick the SPY Elite dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422' );
%     pathname = 'X:\#5 - Data\Human Bone Fracture DCE-FI\001_BoneFI_20190516_083500\01ICG_20190516_083500\';
%     filename = '01ICG_20190516_SPY_FI';

[rgb_file,~]=uigetfile('*.*', 'Pick the White light dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422' );

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
imagesc(d.c.Q(:,:,600));
h0 = drawrectangle('StripeColor','k');
% select rectanglar region of background
h5 = drawrectangle;
bw=createMask(h0);
Ttl={'GC-Contrast','Energy','SumOfSquares','Entropy','Homogeneity','SumAverage',...
    'SumEntropy','SumVairance','DifferenceEntropy','DifferenceVariance','IMFCorrelation1','IMFCorrelation2','Correlation',...
    'Mean','STD','Skewness','Kurtosis','gamma-k','gamma-phi','IM-Contrast'};
real_width=1024;
real_height=768;
savefolder=('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\multiple patient data results\paramatric maps\numlevels\');
m=787; 
I=d.c.Q(:,:,m);
Back=imcrop(I,h5.Position);
Background(i)=nanmean(Back(:));
ROI1=imcrop(I,h0.Position);
% feature extraction on pixel-by-pixel sliding sub-ROIs
[psta_256]=imageFeature(I,bw,256,2,21,1);
psta_256(:,:,20)=(psta_256(:,:,14)-Background(i))/nanstd(Back(:));
save('DHA007 0428 icg02 texture o2f21nl256sub20 TTP+120s.mat','psta_256');

%% PCA and K-means clustering (add new data on exsiting PCA and clustering)
% import organized training dataset
fs_10=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P10 21features noclasses nl256 offset2 TTP-TTP+120s.mat');
fs_12=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P12 21features noclasses nl256 offset2 TTP-TTP+120s.mat');
fs_16=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P16 21features noclasses nl256 offset2 TTP-TTP+120s.mat');
% import testing dataset
test=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHA007\DHA007 0428 icg02 texture o2f21nl256sub20 TTP.mat');
[xx,yy,zz] = size(test);
XTest=[];
% organize testing dataset
for i=1:zz
    XTest=[XTest reshape(test(:,:,i),[xx*yy 1])];
end
XPredictors=normalize(XTest,'range');
Texture=normalize([fs_10(:,1:20);fs_12(:,1:20);fs_16(:,1:20)],'range');
% principle component analysis
[coeff,score,~,~,explained,mu]=pca(Texture);
Xcentered=score*coeff';
X=score(:,1:3);
% apply PCA to testing data
scoreTest95 = (XPredictors-mu)*coeff(:,1:3); 
opts = statset('Display','final');
[idx,C] = kmeans(X,3,'Distance','cityblock',...
    'Replicates',5,'Options',opts);
% assign testing data to existing clusters
[~,idx_test] = pdist2(C,scoreTest95,'euclidean','Smallest',1); 

% 3D scatter of k-means clustering
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
Cleanedtable=array2table(Cleaned,'VariableNames',{'256-GC-Contrast','256-Energy','256-SumOfSquares','256-Entropy','256-Homogeneity','256-SumAverage',...
    '256-SumEntropy','256-SumVairance','256-DifferenceEntropy','256-DifferenceVariance','256-IMFCorrelation1','256-IMFCorrelation2','256-Correlation',...
    'Mean','STD','Skewness','Kurtosis','IM-Contrast','gamma-k','gamma-phi','Class'});
writetable(Cleanedtable,'testing A07-icg02 all pixels 20features 3classes by 3d Clustering TTP+120s offset2nl256 0428.csv');

%% fluorescence intensity extraction of testing dataset
[filename, pathname] = uigetfile('*.*', 'Pick the SPY Elite dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422' );
%
%     pathname = 'X:\#5 - Data\Human Bone Fracture DCE-FI\001_BoneFI_20190516_083500\01ICG_20190516_083500\';
%     filename = '01ICG_20190516_SPY_FI';

[rgb_file,~]=uigetfile('*.*', 'Pick the White light dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422' );
PRE_TIME=10;

% DICOM
info = dicominfo( [pathname filename] );
d.t = linspace(0, (info.NumberOfFrames-1)*(info.FrameTime/1000), info.NumberOfFrames);
dicom_data = dicomread( [pathname filename] );

d.FI = squeeze(double(dicom_data));
d.Q = d.FI - mean( d.FI(:,:,1:find(d.t > PRE_TIME,1,'first')), 3 );
[X,Y,Z] = size(d.Q);
if exist('rgb_file')
    d.RGB = squeeze(dicomread([pathname,rgb_file]));
end
Q_l = reshape(d.Q, [X*Y Z]);
[Q_max_l, I_max_l] = max(Q_l,[],2);
d.Q_max = reshape(Q_max_l,[X Y]);
% line to select the orientation of the image (rotate), should be parallel
% % to what you want the x-axis to be.
imagesc(d.RGB); 
colormap(goodmap('linearL'));axis off;
h2 = imline;
pos2 = getPosition(h2);
theta=atan((pos2(1,2)-pos2(2,2))/(pos2(1,1)-pos2(1,2)));

% rotate image.
d.c.RGB = imrotate(d.RGB,rad2deg(theta),'bilinear');
d.c.Q = zeros(size(d.c.RGB,1),size(d.c.RGB,2),Z);
for i = 1:Z
    d.c.Q(:,:,i) = imrotate(d.Q(:,:,i),rad2deg(theta),'bilinear');
end

I=d.c.Q(:,:,337);
imagesc(I);
axis off;
h0 = drawrectangle('StripeColor','k');
bw=createMask(h0);
ROI1=imcrop(I,h0.Position);

% select frame from time-to-peak plus 0/10/...120s
I2=d.c.Q(:,:,412);
I3=d.c.Q(:,:,449);
I4=d.c.Q(:,:,487);
I5=d.c.Q(:,:,524);
I6=d.c.Q(:,:,562);
I7=d.c.Q(:,:,599);
I8=d.c.Q(:,:,637);
I9=d.c.Q(:,:,674);
I10=d.c.Q(:,:,712);
I11=d.c.Q(:,:,749);
I12=d.c.Q(:,:,787);
I1=d.c.Q(:,:,374);
Texture0=reshape(I,[X*Y 1]);
save('DHA007 icg02 TTP+0s test-FI.mat','-v7','Texture0');
Texture1=reshape(I1,[X*Y 1]);
save('DHA007 icg02 TTP+10s test-FI.mat','-v7','Texture1');
Texture2=reshape(I2,[X*Y 1]);
save('DHA007 icg02 TTP+20s test-FI.mat','-v7','Texture2');
Texture3=reshape(I3,[X*Y 1]);
save('DHA007 icg02 TTP+30s test-FI.mat','-v7','Texture3');
Texture4=reshape(I4,[X*Y 1]);
save('DHA007 icg02 TTP+40s test-FI.mat','-v7','Texture4');
Texture5=reshape(I5,[X*Y 1]);
save('DHA007 icg02 TTP+50s test-FI.mat','-v7','Texture5');
Texture6=reshape(I6,[X*Y 1]);
save('DHA007 icg02 TTP+60s test-FI.mat','-v7','Texture6');
Texture7=reshape(I7,[X*Y 1]);
save('DHA007 icg02 TTP+70s test-FI.mat','-v7','Texture7');
Texture8=reshape(I8,[X*Y 1]);
save('DHA007 icg02 TTP+80s test-FI.mat','-v7','Texture8');
Texture9=reshape(I9,[X*Y 1]);
save('DHA007 icg02 TTP+90s test-FI.mat','-v7','Texture9');
Texture10=reshape(I10,[X*Y 1]);
save('DHA007 icg02 TTP+100s test-FI.mat','-v7','Texture10');
Texture11=reshape(I11,[X*Y 1]);
save('DHA007 icg02 TTP+110s test-FI.mat','-v7','Texture11');
Texture12=reshape(I12,[X*Y 1]);
save('DHA007 icg02 TTP+120s test-FI.mat','-v7','Texture12');

 %% overlay classifer map with surgeons outline image and calculate accuracy/sensitivity
% load outline image
I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007 POST BONE CUT AND POST SOFT TISSUE STRIPPING\exportedStudy-20210422-00030\icg02-RGB.png');
% resize the outline image to be the same as k-means clustering map and FI map 
I_resize=imresize(I,[748 1004]);
I_1=imresize(I,[768 1024]);

imshow(I_resize);
axis off;
h0 = drawfreehand('closed');
bw1=createMask(h0);
imshow(I_1);
axis off;
h = drawfreehand('closed');
bw=createMask(h);
% iterate through every 10 seconds
for i=1:12
    fs_table=readtable(['C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\testing A07-icg02 all pixels 20features 3classes by 3d Clustering TTP offset2nl256 0428.csv']);
    fs=table2array(fs_table);
    selectedn=fs(:,21);
    label_image = reshape(selectedn, [748,1004]);
    ROI1=label_image.*(double(bw1)); % k-means clustering map
    qmax_table=readtable(['C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\DHA007 icg02 TTP+',num2str(i),'s FI-ypredict.csv']);
    qmax=table2array(qmax_table);
    selectedqmax=qmax(:,2);
    label_image_qmax = reshape(selectedqmax, [768,1024]);
    ROI=label_image_qmax.*(double(bw)); % FI map
    
    figure(4*i)
    % overlay outline image with FI map
    ii=imshow(I_1);
    hold on;
    im=imagesc(ROI);
    colormap([1 1 1;0 1 0; 1 1 0;1 0 0]);
    im.AlphaData = 0.4;
    colorbar;
    hold off;
    
    figure(4*i+1)
    % overlay outline image with k-means clustering map
    ii=imshow(I_resize);
    hold on;
    im=imagesc(ROI1);
    colormap([1 1 1;1 1 0; 0 1 0;1 0 0]);
    im.AlphaData = 0.4;
    colorbar;
    hold off;
   
    ypredict_cluster=nonzeros(ROI1);
    ypredict_qmax=nonzeros(ROI);
    yactual_cluster=ones(size(ypredict_cluster,1),1).*1;
    yactual_qmax=ones(size(ypredict_qmax,1),1).*3;
    
    figure(4*i+2)
    % confusion matrix of k-means clustering
    C1=confusionmat(yactual_cluster,ypredict_cluster);
    confusionchart(C1)
    
    figure(4*i+3)
    % confusion matrix of FI
    C2=confusionmat(yactual_qmax,ypredict_qmax);
    confusionchart(C2)
end
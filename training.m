%% feature extraction of training dataset: offset=2,numlevel 256, changing glcm
close all

% Constants
PRE_TIME = 10; %s - time before ICG injection (conservative estimate).

% open DICOM stack
[filename, pathname] = uigetfile('*.*', 'Pick the SPY Elite dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\016_BoneFI_20200227' );
%     pathname = 'X:\#5 - Data\Human Bone Fracture DCE-FI\001_BoneFI_20190516_083500\01ICG_20190516_083500\';
%     filename = '01ICG_20190516_SPY_FI';
[rgb_file,~]=uigetfile('*.*', 'Pick the White light dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\016_BoneFI_20200227' );

% DICOM processing
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

%rotate image.
d.c.RGB = imrotate(d.RGB,rad2deg(theta),'bilinear');
d.c.TTP = imrotate(d.TTP,rad2deg(theta),'bilinear');
d.c.Qmax = imrotate(d.Q_max,rad2deg(theta),'bilinear');
d.c.Q = zeros(size(d.c.RGB,1),size(d.c.RGB,2),Z);
for i = 1:Z
    d.c.Q(:,:,i) = imrotate(d.Q(:,:,i),rad2deg(theta),'bilinear');
end

% select rectanglar region of interest
imagesc(d.c.Q(:,:,500));
h0 = drawrectangle('StripeColor','k');
h5 = drawrectangle; %select rectanglar region of background
bw=createMask(h0);
i=0;
I=zeros(X,Y);
startf=[178:345,358:1024];

% texture analysis on sub-ROIs within all useful frames
for m=startf
    i=i+1;
    I=d.c.Q(:,:,m);
    Back=imcrop(I,h5.Position);
    Background(i)=nanmean(Back(:));
    ROI1=imcrop(I,h0.Position);
    [Subtexture1,posi,N1]=split_image(ROI1,20,20); %split sub-ROIs
    for ii=1:N1
        I1=Subtexture1(:,:,ii);
        low=min(I1(:),[],'all');
        high=max(I1(:),[],'all');
        h=2*iqr(I1,[1 2])*numel(I1(~isnan(I1)))^(-1/3)+0.01;
        bin=ceil(((high-low)/h)/5)+1;
        GLCM1=graycomatrix(I1,'NumLevels',256,'Graylimits',[],'Offset',[0 2;-2 2;-2 0;-2 2],'Symmetric',true); % texture analysis
        comatrix_1=GLCM1(:,:,1);
        comatrix_2=GLCM1(:,:,2);
        comatrix_3=GLCM1(:,:,3);
        comatrix_4=GLCM1(:,:,4);
        comatrix_1(~any(GLCM1(:,:,1),2),:) = [];  %delete rows of all zeros
        comatrix_1(:,~any(GLCM1(:,:,1),1)) = [];  %delete columns of all zeros
        comatrix_2(~any(GLCM1(:,:,2),2),:) = [];  
        comatrix_2(:,~any(GLCM1(:,:,2),1)) = [];  
        comatrix_3(~any(GLCM1(:,:,3),2),:) = [];  
        comatrix_3(:,~any(GLCM1(:,:,3),1)) = [];  
        comatrix_4(~any(GLCM1(:,:,4),2),:) = [];  
        comatrix_4(:,~any(GLCM1(:,:,4),1)) = [];  
        stats_1=struct2cell(GLCM_Features(comatrix_1));
        stats_2=struct2cell(GLCM_Features(comatrix_2));
        stats_3=struct2cell(GLCM_Features(comatrix_3));
        stats_4=struct2cell(GLCM_Features(comatrix_4));
        stats_I1=mean(cell2mat([stats_1,stats_2,stats_3,stats_4]),2);
        stats4=GLCM_Features(GLCM1);
        stats_I1(2) = mean(stats4.energ,'all');
        stats_I1(4) = mean(stats4.entro,'all');
        stats_I1(14)=nanmean(I1(:));
        stats_I1(15)=nanstd(I1(:));
        stats_I1(16)=skewness(I1(:));
        stats_I1(17)=kurtosis(I1(:));
        stats_I1(18)=(stats_I1(14)-Background(i))/nanstd(Back(:));
        temp=gamfit(abs(I1(:)));
        stats_I1(19)=temp(1);
        stats_I1(20)=temp(2);
        Newposi(1,ii)=posi(1,ii)+h0.Position(1);
        Newposi(2,ii)=posi(2,ii)+h0.Position(2);
        Newposi(3,ii)=posi(3,ii);
        Newposi(4,ii)=posi(4,ii);
        stats_I1(21)=nanmean(d.c.Qmax(Newposi(2,ii):(Newposi(2,ii)+Newposi(4,ii)),Newposi(1,ii):(Newposi(1,ii)+Newposi(3,ii))),'all');
        texture1(ii,:)=stats_I1';
        clear GLCM1 comatrix_1 comatrix_2 comatrix_3 comatrix_4 stats1 stats2 stats4 stats_1 stats_2 stats_3 stats_4 stats_I1
    end
if i==1
    Texturedata1=zeros((size(startf,1)),21,N1);
end
for j=1:21
    Texturedata1(i,j,:)=texture1(:,j);
end
disp(i);
end
save('DHP016 0412 icg02 texture o2f21nl256sub20.mat','Texturedata1');
%% organize training dataset
% import training dataset of three training patients (No.1~3)
P16_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP0100\DHP016 0412 icg01 texture o2f21nl256sub20.mat');
P16_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP0100\DHP016 0412 icg02 texture o2f21nl256sub20.mat');
P16_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP0100\DHP016 0412 icg03 texture o2f21nl256sub20.mat');
P10_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP010\DHP010 0412 icg01 texture o2f21nl256sub20.mat');
P10_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP010\DHP010 0412 icg02 texture o2f21nl256sub20.mat');
P10_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP010\DHP010 0412 icg03 texture o2f21nl256sub20.mat');
P12_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP012\DHP012 0412 icg01 texture o2f21nl256sub20.mat');
P12_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP012\DHP012 0412 icg02 texture o2f21nl256sub20.mat');
P12_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP012\DHP012 0412 icg03 texture o2f21nl256sub20.mat');

X101_0=concat(P10_01);
X102_0=concat(P10_02);
X103_0=concat(P10_03);
X121_0=concat(P12_01);
X122_0=concat(P12_02);
X123_0=concat(P12_03);
X161_0=concat(P16_01);
X162_0=concat(P16_02);
X163_0=concat(P16_03);

% add column of frame numbers
f101=[repmat([1:size(P10_01,1)]',size(P10_01,3),1)];
f102=[repmat([1:size(P10_02,1)]',size(P10_02,3),1)];
f103=[repmat([1:size(P10_03,1)]',size(P10_03,3),1)];
f121=[repmat([1:size(P12_01,1)]',size(P12_01,3),1)];
f122=[repmat([1:size(P12_02,1)]',size(P12_02,3),1)];
f123=[repmat([1:size(P12_03,1)]',size(P12_03,3),1)];
f161=[repmat([1:size(P16_01,1)]',size(P16_01,3),1)];
f162=[repmat([1:size(P16_02,1)]',size(P16_02,3),1)];
f163=[repmat([1:size(P16_03,1)]',size(P16_03,3),1)];

% add column of patient ID
X101=[X101_0 repmat([10],size(X101_0,1),1) f101];
X102=[X102_0 repmat([10],size(X102_0,1),1) f102];
X103=[X103_0 repmat([10],size(X103_0,1),1) f103];
X121=[X121_0 repmat([12],size(X121_0,1),1) f121];
X122=[X122_0 repmat([12],size(X122_0,1),1) f122];
X123=[X123_0 repmat([12],size(X123_0,1),1) f123];
X161=[X161_0 repmat([16],size(X161_0,1),1) f161];
X162=[X162_0 repmat([16],size(X162_0,1),1) f162];
X163=[X163_0 repmat([16],size(X163_0,1),1) f163];

% select useful frames
T101=X101(f101<=682&f101>=232,:);
T102=X102(f102<=692&f102>=242,:);
T103=X103(f103<=788&f103>=573,:);
T121=X121(f121<=640&f121>=190,:);
T122=X122(f122<=640&f122>=190,:);
T123=X123(f123<=640&f123>=190,:);
T161=X161(f161<=669&f161>=219,:);
T162=X162(f162<=652&f162>=202,:);
T163=X163(f163<=500&f163>=209,:);

T10=[T101(:,1:22) T101(:,23)-231;T102(:,1:22) T102(:,23)-241;T103(:,1:22) T103(:,23)-572];
T12=[T121(:,1:22) T121(:,23)-189;T122(:,1:22) T122(:,23)-189;T123(:,1:22) T123(:,23)-189];
T16=[T161(:,1:22) T161(:,23)-218;T162(:,1:22) T162(:,23)-201;T163(:,1:22) T163(:,23)-208];

% normalize feature max intensity to ranges [0,1]
Norm10=[T10(:,1:20) normalize(T10(:,21),'range') T10(:,22:23)];
Norm12=[T12(:,1:20) normalize(T12(:,21),'range') T12(:,22:23)];
Norm16=[T16(:,1:20) normalize(T16(:,21),'range') T16(:,22:23)];

save('training P10 21features noclasses nl256 offset2 TTP-TTP+120s.mat','Norm10');
save('training P12 21features noclasses nl256 offset2 TTP-TTP+120s.mat','Norm12');
save('training P16 21features noclasses nl256 offset2 TTP-TTP+120s.mat','Norm16');

%% training dataset: single variable of fluorescence intensity 
load('training P10 21features 3classes obviousROI norescale offset2 TTP-TTP+120s.mat')
load('training P12 21features 3classes obviousROI norescale offset2 TTP-TTP+120s.mat')
load('training P16 21features 3classes obviousROI norescale offset2 TTP-TTP+120s.mat')
data=[T10(:,[14,22:24]);T12(:,[14,22:24]);T16(:,[14,22:24])];
Cleanedtable=array2table(data,'VariableNames',{'Mean','ID','Frame','Class'});
writetable(Cleanedtable,'training 1featureMean 3classes TTP-TTP+120s withtime offset2 0312.csv');
%% PCA and K-means clustering
% import organized training dataset
fs_10=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P10 21features noclasses nl256 offset2 TTP-TTP+120s.mat');
fs_12=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P12 21features noclasses nl256 offset2 TTP-TTP+120s.mat');
fs_16=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P16 21features noclasses nl256 offset2 TTP-TTP+120s.mat');

% rescale each feature to range [0, 1]
Texture=normalize([fs_10(:,1:20);fs_12(:,1:20);fs_16(:,1:20)],'range');

% principle component analysis
[coeff,score,~,~,explained,mu]=pca(Texture);
Xcentered=score*coeff';

% biplot of PCA results
figure(1)
vbls = {'contra','ener','sos','entro','homo','sa','se','sv','de','dv','imoc1',...
    'imoc2','corre','mean','std','skew','kurt','IMcontra','gk','gp'}; % Labels for the variables
biplot(coeff(:,1:3),'VarLabels',vbls);
title('PCA t=TTP-TTP+120s')

% k-means clustering
X=score(:,1:3);
opts = statset('Display','final');
[idx,C] = kmeans(X,3,'Distance','cityblock',...
    'Replicates',5,'Options',opts);

% 3D scatter plot of all data points under the three components
figure(2)
h1=scatter3(X(idx==1,1),X(idx==1,2),X(idx==1,3),36,'y.','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
hold on
h2=scatter3(X(idx==2,1),X(idx==2,2),X(idx==2,3),36,'g.','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
h3=scatter3(X(idx==3,1),X(idx==3,2),X(idx==3,3),36,'r.','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
scatter3(C(1,1),C(1,2),C(1,3),'kx','SizeData',15,'LineWidth',3) 
scatter3(C(2,1),C(2,2),C(2,3),'kx','SizeData',15,'LineWidth',3)
scatter3(C(3,1),C(3,2),C(3,3),'kx','SizeData',15,'LineWidth',3)
hold off
legend('Cluster 1','Cluster 2','Cluster 3',...
       'Location','NW')
title 'Cluster Assignments t=TTP-TTP+120s'
xlabel('Component 1')
ylabel('Component 2')
zlabel('Component 3')

% save the data table and add the column of clustering number
Cleaned=[[fs_10;fs_12;fs_16],idx];
Cleanedtable=array2table(Cleaned,'VariableNames',{'256-GC-Contrast','256-Energy','256-SumOfSquares','256-Entropy','256-Homogeneity','256-SumAverage',...
    '256-SumEntropy','256-SumVairance','256-DifferenceEntropy','256-DifferenceVariance','256-IMFCorrelation1','256-IMFCorrelation2','256-Correlation',...
    'Mean','STD','Skewness','Kurtosis','IM-Contrast','gamma-k','gamma-phi','Qmax','ID','Frame','Class'});
writetable(Cleanedtable,'training 20features 3classes by 3d Clustering TTP-TTP+120s withtime offset2nl256 0328.csv');
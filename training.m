%% feature extraction of training dataset: offset=2,numlevel 256, changing glcm
close all

% Constants
PRE_TIME = 10; %s - time before ICG injection (conservative estimate).

% open DICOM stack
[filename, pathname] = uigetfile('*.*', 'Pick the SPY Elite dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422' );
%     pathname = 'X:\#5 - Data\Human Bone Fracture DCE-FI\001_BoneFI_20190516_083500\01ICG_20190516_083500\';
%     filename = '01ICG_20190516_SPY_FI';
[rgb_file,~]=uigetfile('*.*', 'Pick the White light dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422' );

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
imagesc(d.c.RGB);
h0 = drawrectangle('StripeColor','k');
h5 = drawrectangle; %select rectanglar region of background
bw=createMask(h0);
i=0;
I=zeros(X,Y);
startf=[79:263,317:386,477:626,683:1024];

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
save('DHA07 0916 icg03 texture o2f21nl256sub20 TTPtoEND.mat','Texturedata1');
%% organize training dataset
% import training dataset of seven training patients (No.1~7)
P16_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP0100\DHP016 0412 icg01 texture o2f21nl256sub20.mat');
P16_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP0100\DHP016 0412 icg02 texture o2f21nl256sub20.mat');
P16_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP0100\DHP016 0412 icg03 texture o2f21nl256sub20.mat');
P10_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP010\DHP010 0412 icg01 texture o2f21nl256sub20.mat');
P10_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP010\DHP010 0412 icg02 texture o2f21nl256sub20.mat');
P10_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP010\DHP010 0412 icg03 texture o2f21nl256sub20.mat');
P12_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP012\DHP012 0412 icg01 texture o2f21nl256sub20.mat');
P12_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP012\DHP012 0412 icg02 texture o2f21nl256sub20.mat');
P12_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\DHP012\DHP012 0412 icg03 texture o2f21nl256sub20.mat');

P13_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA09 0916 icg03 texture o2f21nl256sub20 TTPtoEND.mat');
A01_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA07 0916 icg01 texture o2f21nl256sub20 TTPtoEND.mat');
A01_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA07 0916 icg02 texture o2f21nl256sub20 TTPtoEND.mat');
A01_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA07 0916 icg03 texture o2f21nl256sub20 TTPtoEND.mat');
A08_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA08 0916 icg01 texture o2f21nl256sub20 TTPtoEND.mat');
A08_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA08 0916 icg02 texture o2f21nl256sub20 TTPtoEND.mat');
A08_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA08 0916 icg03 texture o2f21nl256sub20 TTPtoEND.mat');
A10_01=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA10 0916 icg01 texture o2f21nl256sub20 TTPtoEND.mat');
A10_02=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA10 0916 icg02 texture o2f21nl256sub20 TTPtoEND.mat');
A10_03=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\classification data\New training\DHA10 0916 icg03 texture o2f21nl256sub20 TTPtoEND.mat');

X101_0=concat(P10_01);
X102_0=concat(P10_02);
X103_0=concat(P10_03);
X121_0=concat(P12_01);
X122_0=concat(P12_02);
X123_0=concat(P12_03);
X161_0=concat(P16_01);
X162_0=concat(P16_02);
X163_0=concat(P16_03);

XP132_0=concat(P13_02);
XA011_0=concat(A01_01);
XA012_0=concat(A01_02);
XA013_0=concat(A01_03);
XA081_0=concat(A08_01);
XA082_0=concat(A08_02);
XA083_0=concat(A08_03);
XA101_0=concat(A10_01);
XA102_0=concat(A10_02);
XA103_0=concat(A10_03);

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

fP132=[repmat([1:size(P13_02,1)]',size(P13_02,3),1)];
fA011=[repmat([1:size(A01_01,1)]',size(A01_01,3),1)];
fA012=[repmat([1:size(A01_02,1)]',size(A01_02,3),1)];
fA013=[repmat([1:size(A01_03,1)]',size(A01_03,3),1)];
fA081=[repmat([1:size(A08_01,1)]',size(A08_01,3),1)];
fA082=[repmat([1:size(A08_02,1)]',size(A08_02,3),1)];
fA083=[repmat([1:size(A08_03,1)]',size(A08_03,3),1)];
fA101=[repmat([1:size(A10_01,1)]',size(A10_01,3),1)];
fA102=[repmat([1:size(A10_02,1)]',size(A10_02,3),1)];
fA103=[repmat([1:size(A10_03,1)]',size(A10_03,3),1)];

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

XP132=[XP132_0 repmat([9],size(XP132_0,1),1) fP132];
XA011=[XA011_0 repmat([7],size(XA011_0,1),1) fA011];
XA012=[XA012_0 repmat([7],size(XA012_0,1),1) fA012];
XA013=[XA013_0 repmat([7],size(XA013_0,1),1) fA013];
XA081=[XA081_0 repmat([8],size(XA081_0,1),1) fA081];
XA082=[XA082_0 repmat([8],size(XA082_0,1),1) fA082];
XA083=[XA083_0 repmat([8],size(XA083_0,1),1) fA083];
XA101=[XA101_0 repmat([1010],size(XA101_0,1),1) fA101];
XA102=[XA102_0 repmat([1010],size(XA102_0,1),1) fA102];
XA103=[XA103_0 repmat([1010],size(XA103_0,1),1) fA103];

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

TP132=XP132(fP132<=450,:);
TA011=XA011(fA011<=450,:);
TA012=XA012(fA012<=450,:);
TA013=XA013(fA013<=450,:);
TA081=XA081(fA081<=450,:);
TA082=XA082(fA082<=450,:);
TA083=XA083(fA083<=450,:);
TA101=XA101(fA101<=450,:);
TA102=XA102(fA102<=450,:);
TA103=XA103(fA103<=450,:);

T10=[T101(:,1:22) T101(:,23)-231;T102(:,1:22) T102(:,23)-241;T103(:,1:22) T103(:,23)-572];
T12=[T121(:,1:22) T121(:,23)-189;T122(:,1:22) T122(:,23)-189;T123(:,1:22) T123(:,23)-189];
T16=[T161(:,1:22) T161(:,23)-218;T162(:,1:22) T162(:,23)-201;T163(:,1:22) T163(:,23)-208];

TP13=TP132;
TA01=[TA011;TA012;TA013];
TA08=[TA081;TA082;TA083];
TA10=[TA101;TA102;TA103];

% normalize feature max intensity to ranges [0,1]
Norm10=[T10(:,1:20) normalize(T10(:,21),'range') T10(:,22:23)];
Norm12=[T12(:,1:20) normalize(T12(:,21),'range') T12(:,22:23)];
Norm16=[T16(:,1:20) normalize(T16(:,21),'range') T16(:,22:23)];

NormP13=[TP13(:,1:20) normalize(TP13(:,21),'range') TP13(:,22:23)];
NormA01=[TA01(:,1:20) normalize(TA01(:,21),'range') TA01(:,22:23)];
NormA08=[TA08(:,1:20) normalize(TA08(:,21),'range') TA08(:,22:23)];
NormA10=[TA10(:,1:20) normalize(TA10(:,21),'range') TA10(:,22:23)];

save('training P10 21features noclasses nl256 offset2 TTP-TTP+120s.mat','Norm10');
save('training P12 21features noclasses nl256 offset2 TTP-TTP+120s.mat','Norm12');
save('training P16 21features noclasses nl256 offset2 TTP-TTP+120s.mat','Norm16');

save('training A09 21features noclasses nl256 offset2 TTP-TTP+120s.mat','NormP13');
save('training A07 21features noclasses nl256 offset2 TTP-TTP+120s.mat','NormA01');
save('training A08 21features noclasses nl256 offset2 TTP-TTP+120s.mat','NormA08');
save('training A10 21features noclasses nl256 offset2 TTP-TTP+120s.mat','NormA10');

%% PCA and kmeans clustering training with weighted pc
% import organized training dataset
data1=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P16 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data2=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P12 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data3=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P13 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data4=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P10 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data5=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A08 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data6=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A09 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data7=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A10 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');

% normalize every features to [0,1] from each patient individually
Texture=[normalize(data1(:,1:20),'range');normalize(data2(:,1:20),'range');normalize(data3(:,1:20),'range');...
    normalize(data4(:,1:20),'range');normalize(data5(:,1:20),'range');normalize(data6(:,1:20),'range');normalize(data7(:,1:20),'range')];

% principle component analysis (PCA)
[coeff,score,~,~,explained,mu]=pca(Texture);
Xcentered=score*coeff';

% k-means clustering (k=3) on the top three PCs weighted by square root of
% their percentages of total variance explained
X=[score(:,1).*realsqrt(round(explained(1)/10)),score(:,2).*realsqrt(round(explained(2)/10)),...
    score(:,3).*realsqrt(round(explained(3)/10))];
opts = statset('Display','final');
[idx,C] = kmeans(X,3,'Replicates',5,'Options',opts);

% 3D scatter plot of training data partitioned by k-means clustering:
% Figure 4(b)
figure(3)
h1=scatter3(X(idx==1,1),X(idx==1,2),X(idx==1,3),36,'y.','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
hold on
h2=scatter3(X(idx==2,1),X(idx==2,2),X(idx==2,3),36,'g.','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
h4=scatter3(X(idx==3,1),X(idx==3,2),X(idx==3,3),36,'r.','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
scatter3(C(1,1),C(1,2),C(1,3),'kx','SizeData',15,'LineWidth',3) 
scatter3(C(2,1),C(2,2),C(2,3),'kx','SizeData',15,'LineWidth',3)
scatter3(C(3,1),C(3,2),C(3,3),'kx','SizeData',15,'LineWidth',3)
hold off
legend('Cluster 1','Cluster 2','Cluster 3',...
       'Location','NW')
title ('Cluster Assignments t=TTP-TTP+120s')
xlabel('Weighted Component 1')
ylabel('Weighted Component 2')
zlabel('Weighted Component 3')

% save the data table and add the column of clustering number
Cleaned=[[data1(:,1:23);data2(:,1:23);data3(:,1:23);data4(:,1:23);data5(:,1:23);data6(:,1:23);data7(:,1:23)],idx];
Cleanedtable=array2table(Cleaned,'VariableNames',{'256-GC-Contrast','256-Energy','256-SumOfSquares','256-Entropy','256-Homogeneity','256-SumAverage',...
    '256-SumEntropy','256-SumVairance','256-DifferenceEntropy','256-DifferenceVariance','256-IMFCorrelation1','256-IMFCorrelation2','256-Correlation',...
    'Mean','STD','Skewness','Kurtosis','IM-Contrast','gamma-k','gamma-phi','Qmax','ID','Frame','Class'});
writetable(Cleanedtable,'training 20features 3classes 7patients round2 by 3d Clustering weighted features TTP-TTP+120s withtime offset2nl256 1014.csv');

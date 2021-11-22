%% label ROIs according to decision outline images
close all

% Constants
PRE_TIME = 10; %s - time before ICG injection (conservative estimate).

% open DICOM stack
[filename, pathname] = uigetfile('*.*', 'Pick the SPY Elite dicom containing the DCE-FI data','C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\' );
%     pathname = 'X:\#5 - Data\Human Bone Fracture DCE-FI\001_BoneFI_20190516_083500\01ICG_20190516_083500\';
%     filename = '01ICG_20190516_SPY_FI';
[rgb_file,~]=uigetfile('*.*', 'Pick the White light dicom containing the DCE-FI data',pathname );

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
width=34;
height=8;
h0 = drawrectangle('StripeColor','k','Position',[80 320 width*20 height*20]);
wait(h0);
Bone=imcrop(d.c.RGB,h0.Position);
[xx,yy,zz]=size(Bone);
figure(2)
imagesc(Bone);
set(gcf,'Position',[0 90 width*20 height*20])
h1=drawfreehand; %devide bone region in three parts
h2=drawfreehand;
h3=drawfreehand;
bw1=createMask(h1);
bw2=createMask(h2);
bw3=createMask(h3);
Label1=1.*ones(xx,yy).*double(bw1);
Label2=1.*ones(xx,yy).*double(bw2);
Label3=2.*ones(xx,yy).*double(bw3);
Label_t=Label1+Label2+Label3;
Label_t(Label_t==0)=1;
Label_shrinked=round(imresize(Label_t,[height width]));
Label_vector=reshape(Label_shrinked',[],1);
save('DHA10 0916 icg01 decision outline labels footoff.mat','Label_vector');

%% add label into training dataset
%import the data after texture analysis from each training patient
data=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P16 21features noclasses nl256 offset2 TTP-TTP+120s.mat');

%import the ROI labels from the above section
label_1=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\DHP16 0916 icg01 decision outline labels footoff.mat');
label_2=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\DHP16 0916 icg02 decision outline labels.mat');
label_3=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\DHP16 0916 icg03 decision outline labels.mat');

%add a column of ROI labels
label_1_allframes=repmat(label_1,1,451); %3rd dimension is frame number from Time-To-Peak to 120s later
label_2_allframes=repmat(label_2,1,451);
label_3_allframes=repmat(label_3,1,292);
label_ready=[reshape(label_1_allframes',[],1);reshape(label_2_allframes',[],1);reshape(label_3_allframes',[],1)];
data_icg01=data(1:size(label_1,1)*451,14);
data_icg02=data(1+size(label_1,1)*451:size(label_1,1)*451+size(label_2,1)*451,14);
data_icg03=data(1+size(label_1,1)*451+size(label_2,1)*451:end,14);

%plot histogram of each icg video for finding the over saturated pixels
figure(1)
subplot(1,3,1)
histogram(data_icg01)
ylim([0 6000]);
title('A08 icg01')
subplot(1,3,2)
histogram(data_icg02)
ylim([0 6000]);
title('A08 icg02')
subplot(1,3,3)
histogram(data_icg03)
ylim([0 6000]);
title('A08 icg03')

%adjust the over saturated pixels by assigning them by the closest largest value 
data_icg01_truncated=remove_saturation(data_icg01,170,'intensity');
%data_icg01_truncated=data_icg01;
data_icg02_truncated=remove_saturation(data_icg02,140,'intensity');
%data_icg02_truncated=data_icg02;
data_icg03_truncated=remove_saturation(data_icg03,140,'intensity');
%data_icg03_truncated=data_icg03;

%plot the histogram after the adjustment
figure(2)
subplot(1,3,1)
histogram(data_icg01_truncated)
ylim([0 6000]);
title('A08 truncated icg01')
subplot(1,3,2)
histogram(data_icg02_truncated)
ylim([0 6000]);
title('A08 truncated icg02')
subplot(1,3,3)
histogram(data_icg03_truncated)
ylim([0 6000]);
title('A08 truncated icg03')

%normalized the data to the range of [0,1]
data_FI=rescale([data_icg01_truncated;data_icg02_truncated;data_icg03_truncated]);

% save the data with only feature of "mean" for thresholding classifier
%data_addlabel=[data_FI,data(:,22:23),label_ready];
%save('training A10 footoff 1featureMean truncatedANDrescaled 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat','data_addlabel');
% save the whole data for supervised learning classifier
data_addlabel=[data_FI,data,label_ready];
save('training P16 footoff 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat','data_addlabel');

%% save as FI classifier training dataset
%organize the dataset with feature "mean" from training patients who have
%regular fluorescent intensities
data1=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P16 footoff 1featureMean truncatedANDrescaled 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data2=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P12 footoff 1featureMean truncatedANDrescaled 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data4=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P10 1featureMean truncatedANDrescaled 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data7=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A10 footoff 1featureMean truncatedANDrescaled 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');

data=[data1;data2;data4;data7];
Cleanedtable=array2table(data,'VariableNames',{'Mean','ID','Frame','Class'});
writetable(Cleanedtable,'training footoff 1featureMean 3classesByDecisionOutlines 4patients truncatedANDrescaled TTP-TTP+120s withtime offset2 1021.csv');

%% save as supervised learning training dataset
% organize the dataset from all training patients
data1=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P16 footoff 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data2=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P12 footoff 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data3=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P13 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data4=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training P10 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data5=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A08 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data6=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A09 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');
data7=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training A10 footoff 21features 3classesByDecisionOutlines nl256 offset2 TTP-TTP+120s.mat');

% save the data table for supervised learning
Cleaned=[data1(:,2:25);data2(:,2:25);data3;data4;data5;data6;data7(:,2:25)];
Cleanedtable=array2table(Cleaned,'VariableNames',{'256-GC-Contrast','256-Energy','256-SumOfSquares','256-Entropy','256-Homogeneity','256-SumAverage',...
    '256-SumEntropy','256-SumVairance','256-DifferenceEntropy','256-DifferenceVariance','256-IMFCorrelation1','256-IMFCorrelation2','256-Correlation',...
    'Mean','STD','Skewness','Kurtosis','IM-Contrast','gamma-k','gamma-phi','Qmax','ID','Frame','Class'});
writetable(Cleanedtable,'training footoff 20features 3classes 7patients round2 by DecisionOutlines TTP-TTP+120s withtime offset2nl256 1014.csv');

%% FI thresholding: find two optimal thresholds according to customized cost functions
% import the FI classifier training dataset
data=readtable('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training footoff 1featureMean 3classesByDecisionOutlines 4patients truncatedANDrescaled TTP-TTP+120s withtime offset2 1021.csv');
data_array=table2array(data);

%plot the histogram: Figure S1(b)
figure(1)
h1=histogram(data_array(data_array(:,4)==1,1));
hold on;
h2=histogram(data_array(data_array(:,4)==2,1));
h3=histogram(data_array(data_array(:,4)==3,1));
h1.FaceColor='g';
h2.FaceColor='y';
h3.FaceColor='r';
h1.EdgeColor='none';
h2.EdgeColor='none';
h3.EdgeColor='none';
hold off;
legend('actual normal','actual semi','actual compromised')

n=size(data_array,1);
yy=1;
y_pred=zeros(n,1);
FNR=ones(100,100);
FPR=ones(100,100);
SB=ones(100,100);
inaccuracy=ones(100,100);
TH_1=zeros(1,100);
TH_2=zeros(1,100);

%exhausted search for optimal thresholds that minimize the cost functions:
%Figure S1(b, c)
for th_1=0:0.01:0.99
    xx=1;
    for th_2=0.01:0.01:1
        if th_2>th_1
        for i=1:n
            if data_array(i,1)<=th_1
                y_pred(i)=3;
            elseif data_array(i,1)>th_2
                y_pred(i)=1;
            else
                y_pred(i)=2;
            end
        end
        C=confusionmat(data_array(:,4),y_pred);
        FNR(xx,yy)=(C(2,1)+C(3,1))/(C(2,1)+C(2,2)+C(2,3)+C(3,1)+C(3,2)+C(3,3)); %cost function FNR
        FPR(xx,yy)=(C(1,2)+C(1,3))/(C(1,1)+C(1,2)+C(1,3)); %cost function FPR
        SB(xx,yy)=(C(1,2))/(C(1,2)+C(2,2)+C(3,2)); %cost function SB
        TH_2(xx)=th_2;
        end
        xx=xx+1;
    end
    TH_1(yy)=th_1;
    yy=yy+1;
end

sum=(FNR+FPR+SB); %summed cost function with equal weights: Figure S1(d)

figure(2)
s1=pcolor(0:0.01:0.99,0.01:0.01:1,FNR);
s1.EdgeColor='none';
colorbar
xlabel('threshold 1')
ylabel('threshold 2')
title('FNR')
figure(3)
s2=pcolor(0:0.01:0.99,0.01:0.01:1,FPR);
s2.EdgeColor='none';
colorbar
xlabel('threshold 1')
ylabel('threshold 2')
title('FPR')
figure(4)
s3=pcolor(0:0.01:0.99,0.01:0.01:1,SB);
s3.EdgeColor='none';
colorbar
xlabel('threshold 1')
ylabel('threshold 2')
title('SB')

figure(5)
s4=pcolor(0:0.01:0.99,0.01:0.01:1,sum);
s4.EdgeColor='none';
colorbar
xlabel('threshold 1')
ylabel('threshold 2')
title('summed=FNR+FPR+SB+inaccuracy')

% find the optimal thresholds
[M,I]=min(sum,[],'all','linear');

%% prepare dataset for python clustermap
% data=readtable('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\training 7p round1 pca 1006.csv');
% data_array=table2array(data);
%data_cordered=data_array(:,[6,8,11,9,3,10,1,7,5,13,4,14,16,12,15,17,18,2,19,20,21:24]);
data_array=[X,idx];
data_array(data_array(:,4)==4,4)=5;
data_array(data_array(:,4)==1,4)=4;
% Cleanedtable=array2table(data_array,'VariableNames',{'f6','f8','f11','f9','f3','f10',...
%     'f1','f7','f5','f13','f4','f14','f16',...
%     'f12','f15','f17','f18','f2','f19','f20','Qmax','ID','Frame','Class'});
% writetable(Cleanedtable,'training 20features 3classes 7patients round2 by 3d Clustering weighted features ordered column TTP-TTP+120s withtime offset2nl256 1014.csv');
Cleanedtable=array2table(data_array,'VariableNames',{'PC1','PC2','PC3','Class'});
writetable(Cleanedtable,'training 7patients round2 4k pca 1014.csv');


                
            
        
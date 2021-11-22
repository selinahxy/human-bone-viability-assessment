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
imagesc(d.c.RGB); 
h0 = drawrectangle('StripeColor','k');
bw=createMask(h0);
% select frame from Time-To-Peak to 120s after
I0=zeros(X,Y,7);
for i=1:7
    frame(i)=round(768+37.5*(i-1));
    if frame(i)<=1024
        I0(:,:,i)=d.Q(:,:,frame(i)).*double(bw);
    end
end
save('DHA01 icg02 TTP+0 to 120s test-FI.mat','I0');
%% overlay with decision outline images and calculate confusion matrix: FI thresholding classifier
% import FI from each testing patient individually
data1=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\DHA07 icg01 TTP+0 to 120s test-FI.mat');
data2=importdata('C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\DHA07 icg02 TTP+0 to 120s test-FI.mat');
data1_bone=nonzeros(data1);
data2_bone=nonzeros(data2);

figure(1)
subplot(1,2,1)
histogram(data1_bone)
title('A07 icg01')
subplot(1,2,2)
histogram(data2_bone)
title('A07 icg02')

%adjust the over saturated pixels by assigning them by the closest largest value 
data1_bone_truncated=remove_saturation(data1_bone,240,'intensity'); %A7
%data1_bone_truncated=remove_saturation(data1_bone,70,'intensity'); %A1
%data1_bone_truncated=data1_bone;
%data2_bone_truncated=remove_saturation(data2_bone,20,'intensity'); 
data2_bone_truncated=data2_bone; %A7,A1
data_bone_truncated=[data1_bone_truncated;data2_bone_truncated];
mini=min(data_bone_truncated,[],'all');
maxi=max(data_bone_truncated,[],'all');

% load outline image
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007 POST BONE CUT AND POST SOFT TISSUE STRIPPING\exportedStudy-20210422-00030\icg02-RGB outlines.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\AMPUTATION\DHA-001 SPY\exportedStudy-20200805-00351\exportedSequence-004_output\exportedSequence-004_sample.jpg');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007PRE BONE CUT\exportedStudy-20210422-00029\icg01-outline.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA001\icg01\icg01-outline.jpg');
I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007PRE BONE CUT\exportedStudy-20210422-00029\icg01_RGB_outlines.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA001\icg01\icg01-RGB_outlines.png');

% resize the outline image to be the same as FI map 
I_1=imresize(I,[768 1024]);
figure(2)
imshow(I_1);
axis off;
h = drawfreehand('closed');
bw=createMask(h);
bad_idx = find(bw==0); 
qmax=zeros(768,1024);

% iterate through every 10 seconds from Time-To-Peak to 120s after in classifying the testing ROIs, using
% the optimal thresholds
for i=1:13
    frame=(data1(:,:,i)-mini).*(1/(maxi-mini));
    frame(bad_idx) = NaN;
    for x=1:768
        for y=1:1024
            if isnan(frame(x,y))
            continue
            end
            if frame(x,y)<=0.07 %if the FI< threshold 1, then the ROI is classified as compromised
                qmax(x,y)=3;
            elseif frame(x,y)>0.31 %if the FI> threshold 2, then the ROI is classified as normal
                qmax(x,y)=1;
            else
                qmax(x,y)=2; %if threshold 1< the FI< threshold 2, then the ROI is classified as semi-normal
            end
        end
    end
               
    ROI=imfill(qmax,8); % hole-filling processing
    filled_qmax=ROI.*(double(bw));
    
    figure(4*i+1)
    % overlay outline image with FI map
    ii=imshow(I_1);
    hold on;
    im=imagesc(filled_qmax);
    colormap([1 1 1;0 1 0;1 1 0]);
    im.AlphaData = 0.4;
    colorbar;
    hold off;
    
     ypredict_qmax=nonzeros(filled_qmax);
     yactual_qmax=ones(size(ypredict_qmax,1),1).*1;
    
    figure(4*i+3)
    % confusion matrix of FI
    C2=confusionmat(yactual_qmax,ypredict_qmax);
    confusionchart(C2)
end
 %% overlay with decision outline images and calculate confusion matrix: LG classifier
% load outline image
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007 POST BONE CUT AND POST SOFT TISSUE STRIPPING\exportedStudy-20210422-00030\icg02-RGB outlines.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\AMPUTATION\DHA-001 SPY\exportedStudy-20200805-00351\exportedSequence-004_output\exportedSequence-004_sample.jpg');
I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA007_20210422\DHA-007PRE BONE CUT\exportedStudy-20210422-00029\icg01-outline.png');
%I=imread('C:\Users\f00349n\Desktop\jaw osteoradionecrosis\patient study\DHA001\icg01\icg01-outline.jpg');

% resize the outline image to be the same as LG map 
I_1=imresize(I,[746 1004]);

imshow(I_1);
axis off;
h = drawfreehand('closed');
bw=createMask(h);

% iterate through every 10 seconds starting from Time-To-Peak to 120s after
for i=[0:7,9:11]
    % LG classifier predictive map
    qmax_table=readtable(['C:\Users\f00349n\Desktop\matlab codes Bone DCE-FI Human Study\DHA007 icg01 TTP+',num2str(i),'0s LG-ypredict.csv']);
    qmax=table2array(qmax_table);
    selectedqmax=qmax(:,2);
    label_image_qmax = reshape(selectedqmax, [746,1004]);
    ROI=imfill(label_image_qmax,8); %hole-filling processing
    filled_qmax=ROI.*(double(bw));
    
    figure(4*i+1)
    % overlay outline image with LG map
    ii=imshow(I_1);
    hold on;
    im=imagesc(filled_qmax);
    colormap([1 1 1;0 1 0;1 1 0;1 0 0]);
    im.AlphaData = 0.4;
    colorbar;
    hold off;
    
     ypredict_qmax=nonzeros(filled_qmax);
     yactual_qmax=ones(size(ypredict_qmax,1),1).*2;
        
    figure(4*i+3)
    % confusion matrix of LG
    C2=confusionmat(yactual_qmax,ypredict_qmax);
    confusionchart(C2)
end

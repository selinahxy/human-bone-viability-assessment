function [pstats] = imageFeature(pmap,mask,nL,D,WS,rescale)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract texture features in pmap
% INPUT:
% pmap = input image
% mask = bad fit mask, same size as pmap
% nL = number of gray levels used to discretize the co-occurrence matrix
% D = displacement vector length
% WS = length of square neighborhood size, within which compute texture and
% first order statistics (must be odd, pixel of interest centered)
% rescale = 1 for rescaling GLCM by deleting cells of zeros, =0 for no
% rescaling
% OUTPUT:
% pstats = texture features per WS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masked values set to NaN prior to texture/shape computation
map_sz = size(pmap);
bad_idx = find(mask==0); pmap(bad_idx) = NaN;
% Pre-allocate memory
pstats= zeros(map_sz(1)-WS-1,map_sz(2)-WS-1,19);
% Co-occurence matrix representation of local texture
warning off;
direction = [0 D; -D D; -D 0; -D D];% [0,1]=0deg, [-1,1]=45deg, [-1,0]=90deg, [-1,-1]=135deg;
% Compute texture features in window per pixel
for ri = 0.5*(WS+1):map_sz(1)-0.5*(WS-1)
    for ci = 0.5*(WS+1):map_sz(2)-0.5*(WS-1)
        if isnan(pmap(ri,ci))
            continue
        end
        % Data in window per parameter
        wdata = pmap((ri - 0.5*(WS-1)):(ri+0.5*(WS-1)),...
            (ci - 0.5*(WS-1)):(ci+0.5*(WS-1))); % data in window
        % Compute second order statistics for new pixel
        % (Assuming texture primitives are rotationally invariant,
        % averaging over all angles)
        comatrix1 = graycomatrix(wdata,'NumLevels',nL,'Offset',direction,'Symmetric',true,'GrayLimits',[]);
        if rescale==0
            comatrix=comatrix1;
            stats1=GLCM_Features(comatrix);
            pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),1:13)=structfun(@mean,stats1);
        end
        if rescale==1
            comatrix_1=comatrix1(:,:,1);
            comatrix_2=comatrix1(:,:,2);
            comatrix_3=comatrix1(:,:,3);
            comatrix_4=comatrix1(:,:,4);
            comatrix_1(~any(comatrix1(:,:,1),2),:) = [];  %delete rows with all zeros
            comatrix_1(:,~any(comatrix1(:,:,1),1)) = [];  %delete columns with all zeros
            comatrix_2(~any(comatrix1(:,:,2),2),:) = [];
            comatrix_2(:,~any(comatrix1(:,:,2),1)) = [];
            comatrix_3(~any(comatrix1(:,:,3),2),:) = [];
            comatrix_3(:,~any(comatrix1(:,:,3),1)) = [];
            comatrix_4(~any(comatrix1(:,:,4),2),:) = [];
            comatrix_4(:,~any(comatrix1(:,:,4),1)) = [];
            stats_1=struct2cell(GLCM_Features(comatrix_1));
            stats_2=struct2cell(GLCM_Features(comatrix_2));
            stats_3=struct2cell(GLCM_Features(comatrix_3));
            stats_4=struct2cell(GLCM_Features(comatrix_4));
            pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),1:13)=mean(cell2mat([stats_1,stats_2,stats_3,stats_4]),2);
            stats4=GLCM_Features(comatrix1);
            pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),2) = mean(stats4.energ,'all');
            pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),4) = mean(stats4.entro,'all');
        end
        pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),14)=nanmean(wdata(:));
        pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),15)=nanstd(double(wdata(:)));
        pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),16)=skewness(wdata(:));
        pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),17)=kurtosis(wdata(:));
        temp=gamfit(abs(wdata(:)));
        pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),18)=temp(1);
        pstats((ri-0.5*(WS+1)+1),(ci-0.5*(WS+1)+1),19)=temp(2);
    end
end

end
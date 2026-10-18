tic
close all;clear all;clc;
%%

rGroup=[9]; % [4 7 9] 9
rSubject=1;
rBlock=52;
rSegment=1;
rEmbedding=1:7; % 1:7 3

fs=1000;
n_code=3;
nChannel=59;

nEmbedding=7;
nRangeRate=1.5; % brain radius=1

figureNumSpace=[2]; % 1-3: EEG, LORETAact, LORETAcloud
rangeVlim=[-30,30];

nCloud=10;
range_cloud=[0,0.2];
thr_LORETA=0;
thr_cloud=0;

alphaLORETA=0.5;
alphaCloud=0.5;
grayRate=0.5;

LineWidthLine=1;
markerSizeLORETA=2;
markerSizeLORETA_scatter=2; % 2 0
markerSizeCloud=2;
markerSizeCloud_scatter=10; % 10 0

figureSize={1;[100,500,500,500]};
rRange=1.2;
nColor=100;
Rotation=60;
Tumbling=36; % 16 36
%%

path_LORETAcloud='D:\test_matlab\project_EEG\data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\LORETAcloud\';
path_toolbox={
    'D:\test_matlab\toolbox\eegGA\'
    };
%%

nPath_toolbox=size(path_toolbox,1);
for r=1:nPath_toolbox
    addpath(genpath(path_toolbox{r,1}));
end
%%

coordinate=[];
coordinate_name=[];
load(['scalp(' num2str(nChannel) ').mat']);
for r_group=rGroup
    
    path_LORETAcloud_group=[path_LORETAcloud 'group_' num2code(r_group,n_code) '\'];
    D=dir([path_LORETAcloud_group '*.mat']);
    nSubject=length(D);
    
    if rSubject==0
        rSubject_ture=1:nSubject;
    else
        rSubject_ture=rSubject;
    end
    for r_subject=rSubject_ture
        
        path_LORETAcloud_group_temp=[path_LORETAcloud_group D(r_subject).name];
        
        all_EEG=[];
        all_activation=[];
        all_cloud=[];
        coords=[];
        coords_norm=[];
        load(path_LORETAcloud_group_temp)
        
        nBlock=size(all_EEG,1);
        nCoords=size(coords,1);
        
        if rBlock==0
            rBlock_ture=1:nBlock;
        else
            rBlock_ture=rBlock;
        end
        for r_block=rBlock_ture
            
            nSegment=size(all_EEG{r_block,1},2);
            
            if rSegment==0
                rSegment_ture=1:nSegment;
            else
                rSegment_ture=rSegment;
            end
            for r_segment=rSegment_ture
                
                if rEmbedding==0
                    rEmbedding_ture=1:nEmbedding;
                else
                    rEmbedding_ture=rEmbedding;
                end
                for r_embedding=rEmbedding_ture
                    
                    temp_EEG=all_EEG{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_activation=all_activation{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_cloud=all_cloud{r_block,1}{1,r_segment}{r_embedding,1};
                    %% EEG
                    
                    if ismember(1,figureNumSpace)
                        figure
                        if figureSize{1,1}==1
                            set(gcf,'position',figureSize{2,1});
                        end
                        
                        nT=size(temp_EEG,2);
                        t=(0:nT-1)/fs;
                        xLim=[t(1),t(end)];
                        
                        for rChannel=1:nChannel
                            
                            subplot(nChannel,1,rChannel);
                            plot(t,real(temp_EEG(rChannel,:)),'LineWidth',LineWidthLine);
                            text(-diff(xLim)/10,0,coordinate_name{rChannel,1},'fontWeight','bold');
                            
                            xlim(xLim);
                            ylim(rangeVlim);
                            
                            axis off;
                        end
                    end
                    %% scalp LORETAact
                    
                    if ismember(2,figureNumSpace)
                        figure
                        if figureSize{1,1}==1
                            set(gcf,'position',figureSize{2,1});
                        end
                        
                        scalp_outline;
                        
                        temp_colormap=jet(nColor);
                        for rChannel=1:nChannel
                            
                            hold on;
                            plot3(coordinate(rChannel,1),coordinate(rChannel,2),0,'o','MarkerSize',4,'MarkerFaceColor','k','MarkerEdgeColor','k');
                        end
                        
                        norm_activation=temp_activation/max(temp_activation);
                        for rCoords=1:nCoords
                            
                            temp_norm=norm_activation(1,rCoords);
                            rColor_min=min([(nColor*temp_norm),nColor]);
                            rColor_max=max([1,rColor_min]);
                            rColor=floor(rColor_max);
                            
                            hold on;
                            if markerSizeLORETA_scatter>0
                                if temp_norm>thr_LORETA
                                    scatter3(coords_norm(rCoords,1),coords_norm(rCoords,2),coords_norm(rCoords,3),markerSizeLORETA_scatter,'o','MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:),'MarkerFaceAlpha',alphaLORETA,'MarkerEdgeAlpha',alphaLORETA);
                                else
                                    scatter3(coords_norm(rCoords,1),coords_norm(rCoords,2),coords_norm(rCoords,3),markerSizeLORETA_scatter,'o','MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3),'MarkerFaceAlpha',alphaLORETA,'MarkerEdgeAlpha',alphaLORETA);
                                end
                            else
                                if temp_norm>thr_LORETA
                                    plot3(coords_norm(rCoords,1),coords_norm(rCoords,2),coords_norm(rCoords,3),'o','MarkerSize',markerSizeLORETA,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                                else
                                    plot3(coords_norm(rCoords,1),coords_norm(rCoords,2),coords_norm(rCoords,3),'o','MarkerSize',markerSizeLORETA,'MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3));
                                end
                            end
                        end
                        
                        caxis([0,1]);
                        axis([-rRange rRange -rRange rRange]);
                        axis equal;
                        axis off;
                        view(Rotation,Tumbling);
                    end
                    %% scalp LORETAcloud
                    
                    if ismember(3,figureNumSpace)
                        figure
                        if figureSize{1,1}==1
                            set(gcf,'position',figureSize{2,1});
                        end
                        
                        scalp_outline;
                        
                        temp_colormap=jet(nColor);
                        for r1=1:nChannel
                            
                            hold on;
                            plot3(coordinate(r1,1),coordinate(r1,2),0,'o','MarkerSize',4,'MarkerFaceColor','k','MarkerEdgeColor','k');
                        end
                        
                        xCloud=linspace(-nRangeRate,nRangeRate,2*nCloud);
                        yCloud=linspace(-nRangeRate,nRangeRate,2*nCloud);
                        zCloud=linspace(-nRangeRate,0,nCloud);
                        
                        r0_1=0;
                        for r1=xCloud
                            
                            r0_1=r0_1+1;
                            r0_2=0;
                            for r2=yCloud
                                
                                r0_2=r0_2+1;
                                r0_3=0;
                                for r3=zCloud
                                    
                                    r0_3=r0_3+1;
                                    
                                    temp_v=temp_cloud(r0_1,r0_2,r0_3);
                                    rColor_min=min([(nColor*(temp_v-range_cloud(1))/(diff(range_cloud))),nColor]);
                                    rColor_max=max([1,rColor_min]);
                                    rColor=floor(rColor_max);
                                    
                                    hold on;
                                    if markerSizeCloud_scatter>0
                                        if temp_v>thr_cloud
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:),'MarkerFaceAlpha',alphaCloud,'MarkerEdgeAlpha',alphaCloud);
                                        else
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3),'MarkerFaceAlpha',alphaCloud,'MarkerEdgeAlpha',alphaCloud);
                                        end
                                    else
                                        if temp_v>thr_cloud
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                                        else
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3));
                                        end
                                    end
                                end
                            end
                        end
                        
                        caxis(range_cloud);
                        axis([-rRange rRange -rRange rRange]);
                        axis equal;
                        axis off;
                        view(Rotation,Tumbling);
                    end
                end
            end
        end
    end
end
%%

for r=1:nPath_toolbox
    rmpath(genpath(path_toolbox{r,1}));
end
toc
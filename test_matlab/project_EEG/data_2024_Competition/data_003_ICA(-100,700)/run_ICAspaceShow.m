tic
close all;clear all;clc;
%%

rGroup=[1];
rSubject=3;
rBlock=6;
rSegment=1;
rEmbedding=7;

fs=250;
n_code=3;
nChannel=22;
nComponent=nChannel;

nEmbedding=7;
nRangeRate=1.5; % brain radius=1

figureNumSpace=[1:3]; % 1-6: EEG, Wchannel_abs, Wchannel_norm, scalp power, scalp bregma, scalp loss
figureNumCloud=[0]; % 1-3: scalp power, scalp bregma, scalp loss
rangeVlim=[-20,20];
rangeWchannel_abs=[0,50];
rangeWchannel_norm=[0,0.2];
rangePower=[0,300];
rangeBregma=[0,20];
rangeLoss=[0,20];
shift_coordinate=1;

nCloud=10;
rangePower_cloud=[0,300];
rangeBregma_cloud=[0,10];
rangeLoss_cloud=[0,20];
thrPower=-1;
thrBregma=0;
thrLoss=-1;

faceAlpha=0.5;
grayRate=0.8;

lineWidth=2;
markerSizeSpace=8;
markerSizeCloud=2;
markerSizeCloud_scatter=2;

figureSize={1;[100,500,500,500]};
rRange=1.2;
nColor=100;
Rotation=25;
Tumbling=20;
%%

path_ICAspace='D:\test_matlab\project_EEG\data_2024_Competition\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)\';
path_ICAcloud='D:\test_matlab\project_EEG\data_2024_Competition\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
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
    
    path_ICAspace_group=[path_ICAspace 'group_' num2code(r_group,n_code) '\'];
    D=dir([path_ICAspace_group '*.mat']);
    nSubject=length(D);
    
    if rSubject==0
        rSubject_ture=1:nSubject;
    else
        rSubject_ture=rSubject;
    end
    for r_subject=rSubject_ture
        
        path_ICAspace_group_temp=[path_ICAspace_group D(r_subject).name];
        
        all_EEG=[];
        all_Wchannel_ori=[];
        all_Wchannel=[];
        all_power=[];
        all_bregma=[];
        all_loss=[];
        all_X=[];
        load(path_ICAspace_group_temp)
        nBlock=size(all_EEG,1);
        
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
                    temp_Wchannel_ori=all_Wchannel_ori{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_Wchannel=all_Wchannel{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_power=all_power{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_bregma=all_bregma{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_loss=all_loss{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_X=all_X{r_block,1}{1,r_segment}{r_embedding,1};
                    
                    temp_X=shift_coordinate*temp_X;
                    %% EEG
                    
                    if ismember(1,figureNumSpace)
                        figure
                        if figureSize{1,1}==1
                            set(gcf,'position',figureSize{2,1});
                        end
                        
                        nT=size(temp_EEG,2);
                        t=(0:nT-1)/fs;
                        xLim=[t(1),t(end)];
                        
                        for r1=1:nChannel
                            
                            subplot(nChannel,1,r1);
                            plot(t,real(temp_EEG(r1,:)));
                            text(-diff(xLim)/10,0,coordinate_name{r1,1},'fontWeight','bold');
                            
                            xlim(xLim);
                            ylim(rangeVlim);
                            
                            axis off;
                        end
                    end
                    %% Wchannel_abs
                    
                    if ismember(2,figureNumSpace)
                        figure
                        if figureSize{1,1}==1
                            set(gcf,'position',figureSize{2,1});
                        end
                        
                        imagesc(abs(temp_Wchannel_ori));
                        caxis(rangeWchannel_abs);
                        axis equal;
                        axis off;
                    end
                    
                    %% Wchannel_norm
                    
                    if ismember(3,figureNumSpace)
                        figure
                        if figureSize{1,1}==1
                            set(gcf,'position',figureSize{2,1});
                        end
                        
                        temp_interval=1;
                        temp_radius=0.4;
                        temp_res=100;
                        
                        theta=0:2*pi/temp_res:2*pi;
                        A=temp_radius*cos(theta);
                        B=temp_radius*sin(theta);
                        
                        colormapWchannel=jet(nColor);
                        for r1=1:nChannel
                            for r2=1:nComponent
                                
                                temp_temp_Wchannel=temp_Wchannel(r1,r2);
                                rColor_min=min([(nColor*(temp_temp_Wchannel-rangeWchannel_norm(1))/(diff(rangeWchannel_norm))),nColor]);
                                rColor_max=max([1,rColor_min]);
                                rColor=floor(rColor_max);
                                
                                plot(temp_interval*r2+A,-temp_interval*r1+B,'Color',colormapWchannel(rColor,:),'LineWidth',lineWidth);
                                hold on;
                                
                                temp_temp_Wchannel_ori=temp_Wchannel_ori(r1,r2);
                                temp_phase=atan(imag(temp_temp_Wchannel_ori)/real(temp_temp_Wchannel_ori));
                                
                                plot((temp_interval*r2+[0,temp_radius*cos(temp_phase)]),(-temp_interval*r1+[0,temp_radius*sin(temp_phase)]),'k','LineWidth',lineWidth);
                                hold on;
                            end
                        end
                        
                        axis equal;
                        axis off;
                    end
                    %% scalp power
                    
                    if ismember(4,figureNumSpace)
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
                        for r1=1:nComponent
                            
                            rColor_min=min([(nColor*(temp_power(r1)-rangePower(1))/(diff(rangePower))),nColor]);
                            rColor_max=max([1,rColor_min]);
                            rColor=floor(rColor_max);
                            
                            hold on;
                            plot3(temp_X(r1,1),temp_X(r1,2),temp_X(r1,3),'o','MarkerSize',markerSizeSpace,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                        end
                        
                        caxis(rangePower);
                        axis([-rRange rRange -rRange rRange]);
                        axis equal;
                        axis off;
                        view(Rotation,Tumbling);
                    end
                    %% scalp bregma
                    
                    if ismember(5,figureNumSpace)
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
                        for r1=1:nComponent
                            
                            rColor_min=min([(nColor*(temp_bregma(r1)-rangeBregma(1))/(diff(rangeBregma))),nColor]);
                            rColor_max=max([1,rColor_min]);
                            rColor=floor(rColor_max);
                            
                            hold on;
                            plot3(temp_X(r1,1),temp_X(r1,2),temp_X(r1,3),'o','MarkerSize',markerSizeSpace,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                        end
                        
                        caxis(rangeBregma);
                        axis([-rRange rRange -rRange rRange]);
                        axis equal;
                        axis off;
                        view(Rotation,Tumbling);
                    end
                    %% scalp loss
                    
                    if ismember(6,figureNumSpace)
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
                        for r1=1:nComponent
                            
                            rColor_min=min([(nColor*(temp_loss(r1)-rangeLoss(1))/(diff(rangeLoss))),nColor]);
                            rColor_max=max([1,rColor_min]);
                            rColor=floor(rColor_max);
                            
                            hold on;
                            plot3(temp_X(r1,1),temp_X(r1,2),temp_X(r1,3),'o','MarkerSize',markerSizeSpace,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                        end
                        
                        caxis(rangeLoss);
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

xCloud=linspace(-nRangeRate,nRangeRate,2*nCloud);
yCloud=linspace(-nRangeRate,nRangeRate,2*nCloud);
zCloud=linspace(-nRangeRate,0,nCloud);
for r_group=rGroup
    
    path_ICAcloud_group=[path_ICAcloud 'group_' num2code(r_group,n_code) '\'];
    D=dir([path_ICAcloud_group '*.mat']);
    nSubject=length(D);
    
    if rSubject==0
        rSubject_ture=1:nSubject;
    else
        rSubject_ture=rSubject;
    end
    for r_subject=rSubject_ture
        
        path_ICAspace_group_temp=[path_ICAcloud_group D(r_subject).name];
        
        all_power_cloud=[];
        all_bregma_cloud=[];
        all_loss_cloud=[];
        load(path_ICAspace_group_temp);
        nBlock=size(all_bregma_cloud,1);
        
        if rBlock==0
            rBlock_ture=1:nBlock;
        else
            rBlock_ture=rBlock;
        end
        for r_block=rBlock_ture
            
            nSegment=size(all_bregma_cloud{r_block,1},2);
            
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
                    
                    temp_power_cloud=all_power_cloud{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_bregma_cloud=all_bregma_cloud{r_block,1}{1,r_segment}{r_embedding,1};
                    temp_loss_cloud=all_loss_cloud{r_block,1}{1,r_segment}{r_embedding,1};
                    %% scalp power
                    
                    if ismember(1,figureNumCloud)
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
                        
                        r0_1=0;
                        for r1=xCloud
                            
                            r0_1=r0_1+1;
                            r0_2=0;
                            for r2=yCloud
                                
                                r0_2=r0_2+1;
                                r0_3=0;
                                for r3=zCloud
                                    
                                    r0_3=r0_3+1;
                                    
                                    temp_v=temp_power_cloud(r0_1,r0_2,r0_3);
                                    rColor_min=min([(nColor*(temp_v-rangePower_cloud(1))/(diff(rangePower_cloud))),nColor]);
                                    rColor_max=max([1,rColor_min]);
                                    rColor=floor(rColor_max);
                                    
                                    hold on;
                                    if markerSizeCloud_scatter>0
                                        if temp_v>thrPower
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:),'MarkerFaceAlpha',faceAlpha,'MarkerEdgeAlpha',faceAlpha);
                                        else
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3),'MarkerFaceAlpha',faceAlpha,'MarkerEdgeAlpha',faceAlpha);
                                        end
                                    else
                                        if temp_v>thrPower
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                                        else
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3));
                                        end
                                    end
                                end
                            end
                        end
                        
                        caxis(rangePower_cloud);
                        axis([-rRange rRange -rRange rRange]);
                        axis equal;
                        axis off;
                        view(Rotation,Tumbling);
                    end
                    %% scalp bregma
                    
                    if ismember(2,figureNumCloud)
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
                        
                        r0_1=0;
                        for r1=xCloud
                            
                            r0_1=r0_1+1;
                            r0_2=0;
                            for r2=yCloud
                                
                                r0_2=r0_2+1;
                                r0_3=0;
                                for r3=zCloud
                                    
                                    r0_3=r0_3+1;
                                    
                                    temp_v=temp_bregma_cloud(r0_1,r0_2,r0_3);
                                    rColor_min=min([(nColor*(temp_v-rangeBregma_cloud(1))/(diff(rangeBregma_cloud))),nColor]);
                                    rColor_max=max([1,rColor_min]);
                                    rColor=floor(rColor_max);
                                    
                                    hold on;
                                    if markerSizeCloud_scatter>0
                                        if temp_v>thrBregma
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:),'MarkerFaceAlpha',faceAlpha,'MarkerEdgeAlpha',faceAlpha);
                                        else
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3),'MarkerFaceAlpha',faceAlpha,'MarkerEdgeAlpha',faceAlpha);
                                        end
                                    else
                                        if temp_v>thrBregma
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                                        else
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3));
                                        end
                                    end
                                end
                            end
                        end
                        
                        caxis(rangeBregma_cloud);
                        axis([-rRange rRange -rRange rRange]);
                        axis equal;
                        axis off;
                        view(Rotation,Tumbling);
                    end
                    %% scalp loss
                    
                    if ismember(3,figureNumCloud)
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
                        
                        r0_1=0;
                        for r1=xCloud
                            
                            r0_1=r0_1+1;
                            r0_2=0;
                            for r2=yCloud
                                
                                r0_2=r0_2+1;
                                r0_3=0;
                                for r3=zCloud
                                    
                                    r0_3=r0_3+1;
                                    
                                    temp_v=temp_loss_cloud(r0_1,r0_2,r0_3);
                                    rColor_min=min([(nColor*(temp_v-rangeLoss_cloud(1))/(diff(rangeLoss_cloud))),nColor]);
                                    rColor_max=max([1,rColor_min]);
                                    rColor=floor(rColor_max);
                                    
                                    hold on;
                                    if markerSizeCloud_scatter>0
                                        if temp_v>thrLoss
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:),'MarkerFaceAlpha',faceAlpha,'MarkerEdgeAlpha',faceAlpha);
                                        else
                                            scatter3(r1,r2,r3,markerSizeCloud_scatter,'o','MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3),'MarkerFaceAlpha',faceAlpha,'MarkerEdgeAlpha',faceAlpha);
                                        end
                                    else
                                        if temp_v>thrLoss
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                                        else
                                            plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3));
                                        end
                                    end
                                end
                            end
                        end
                        
                        caxis(rangeLoss_cloud);
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
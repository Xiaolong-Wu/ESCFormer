tic
close all;clear all;clc;
%%

rGroup=[4 7 9 13 16 18 22 25 27]; % [4 7 9 13 16 18 22 25 27]

fs=1000;
n_code=3;

nChannel=59;
nComponent=nChannel;

nEmbedding=7;
nEmbedding_olr=0.5;
Band=[2,45];

type='kurtosis';
flag=0;
normalization=1; % 0 1
normalizationCode=1; % 1 2

nPixel=7;
nRangeRate=1.5; % brain radius=1
rangeBregma=[0,20];

A=[0,0,1,0;0,0,0,-1];
b=[0;-1];
Aeq=[];
beq=[];
lb=[];
ub=[];
nonlcon=[];

% 'Display': 'iter' 'final' 'off'
% 'UseParallel': false
options=optimoptions('fmincon','Display','off','UseParallel',false);

nCloud=10;
rangeBregma_cloud=rangeBregma;
thrBregma=5;

figureShow=0; % 6
figureShow_cloud=0; % 6

markerSizeSpace=8;
markerSizeCloud=2;
figureSize={1;[100,500,500,500]};
rRange=1.2;
nColor=100;
Rotation=25;
Tumbling=20;
%%

path_temp=[pwd '\'];
path_block='D:\test_matlab\project_EEG\data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\block\';
if normalization==1
    path_ICAspace=['D:\test_matlab\project_EEG\data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(' num2str(normalizationCode) ')\'];
    path_ICAcloud=['D:\test_matlab\project_EEG\data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(' num2str(normalizationCode) ')\'];
else
    path_ICAspace=['D:\test_matlab\project_EEG\data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm0(' num2str(normalizationCode) ')\'];
    path_ICAcloud=['D:\test_matlab\project_EEG\data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm0(' num2str(normalizationCode) ')\'];
end
path_toolbox={
    'D:\test_matlab\toolbox\eegGA\';
    'D:\test_matlab\toolbox\pca_ica\'
    };
%%

nPath_toolbox=size(path_toolbox,1);
for r=1:nPath_toolbox
    addpath(genpath(path_toolbox{r,1}));
end
%%

global coordinate
global temp_Wchannel_s

coordinate=[];
load(['scalp(' num2str(nChannel) ').mat']);

nInterval=2*nRangeRate/nPixel;
nGroup=length(rGroup);
for r_group=rGroup
    
    path_group=[path_block 'group_' num2code(r_group,n_code) '\'];
    D=dir([path_group '*.mat']);
    
    path_ICAspace_group=[path_ICAspace 'group_' num2code(r_group,n_code) '\'];
    if ~exist(path_ICAspace_group,'dir')
        mkdir(path_ICAspace_group)
    end
    
    nSubject=length(D);
    for r_subject=1:nSubject
        
        temp_name=D(r_subject).name;
        path_group_temp=[path_group temp_name];
        path_ICAspace_group_temp=[path_ICAspace_group temp_name];
        if exist(path_ICAspace_group_temp,'file')
            disp([path_ICAspace_group_temp ' exist, skip.']);
        else
            all_EEG=[];
            all_Wchannel_ori=[];
            all_Wchannel=[];
            all_power=[];
            all_bregma=[];
            all_loss=[];
            all_X=[];
            
            block=[];
            load(path_group_temp);
            nBlock=size(block,1);
            for r_block=1:nBlock
                
                path_ICAspace_group_temp_block=[path_ICAspace_group '\block\'];
                if ~exist(path_ICAspace_group_temp_block,'dir')
                    mkdir(path_ICAspace_group_temp_block)
                end
                
                path_ICAspace_group_temp_block_temp=[path_ICAspace_group_temp_block temp_name(1:end-4) '_block_' num2code(r_block,n_code) '.mat'];
                if exist(path_ICAspace_group_temp_block_temp,'file')
                    disp([path_ICAspace_group_temp_block_temp ' exist, skip.']);
                else
                    nSegment=size(block{r_block,1},2);
                    for r_segment=1:nSegment
                        
                        disp(['ICAspace: rBlock/nBlock: ' num2str(r_block) '/' num2str(nBlock) ', rSegment/nSegment: ' num2str(r_segment) '/' num2str(nSegment)]);
                        EEG_segment=block{r_block,1}{1,r_segment};
                        
                        % temp_nT+(nEmbedding-1)*(1-nEmbedding_olr)*temp_nT=nT
                        nT=size(EEG_segment,2);
                        temp_nT=floor(nT/(1+(nEmbedding-1)*(1-nEmbedding_olr)));
                        
                        for r1=1:nChannel
                            [~,temp_PS]=wave_PS(EEG_segment(r1,:),fs,Band,[0,0],'flat');
                            EEG_segment(r1,:)=temp_PS{1,1}(1,:);
                        end
                        EEG_segment_HT=hilbert(EEG_segment')';
                        
                        for r_embedding=1:nEmbedding
                            
                            first=1+(r_embedding-1)*(1-nEmbedding_olr)*temp_nT;
                            last=first-1+temp_nT;
                            
                            temp_EEG_segment_HT=gpuArray(EEG_segment_HT(:,first:last)); % gpu
                            
                            % Zr = T \ W' * Zica + repmat(mu,1,n)
                            [Zica,W,T,mu]=fastICA(temp_EEG_segment_HT,nComponent,type,flag);
                            Wcomponent=T\W';
                            Wchannel=W'\T;
                            
                            if normalization==1
                                Wnorm=gpuArray(zeros(nComponent,nComponent)); % gpu
                                for r1=1:nComponent
                                    
                                    temp_norm=sum(abs(Zica(r1,:)));
                                    
                                    Zica(r1,:)=Zica(r1,:)/temp_norm;
                                    Wnorm(r1,r1)=temp_norm;
                                end
                                
                                % Zr = T \ W' * Wnorm * Zica + repmat(mu,1,n)
                                Wcomponent=Wcomponent*Wnorm;
                                Wchannel=Wnorm\Wchannel;
                            end
                            
                            temp_EEG=Wcomponent*Zica;
                            temp_Wchannel_ori=Wchannel;
                            temp_Wchannel=temp_Wchannel_ori;
                            temp_power=gpuArray(zeros(nComponent,1)); % gpu
                            temp_bregma=zeros(nComponent,1);
                            temp_loss=zeros(nComponent,1);
                            temp_X=zeros(nComponent,3);
                            for r1=1:nComponent
                                
                                temp_power(r1,1)=sum(abs(Wchannel(r1,:)));
                                temp_Wchannel(r1,:)=abs(Wchannel(r1,:))/temp_power(r1,1);
                            end
                            
                            for r1=1:nComponent
                                temp_Wchannel_s=gather(temp_Wchannel(r1,:)); % gpu gather
                                
                                xb=[];
                                fval=inf;
                                for r2=-nRangeRate:nInterval:nRangeRate
                                    for r3=-nRangeRate:nInterval:nRangeRate
                                        for r4=-nRangeRate:nInterval:0
                                            
                                            x0=[r2,r3,r4,0];
                                            [temp_xb,temp_fval]=fmincon(@ICAspace,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
                                            
                                            if temp_fval<fval
                                                fval=temp_fval;
                                                xb=temp_xb;
                                            end
                                        end
                                    end
                                end
                                
                                temp_bregma(r1,1)=xb(4);
                                temp_loss(r1,1)=fval;
                                temp_X(r1,:)=xb(1:3);
                            end
                            
                            all_EEG_block{1,r_segment}{r_embedding,1}=gather(temp_EEG); % gpu gather
                            all_Wchannel_ori_block{1,r_segment}{r_embedding,1}=gather(temp_Wchannel_ori); % gpu gather
                            all_Wchannel_block{1,r_segment}{r_embedding,1}=gather(temp_Wchannel); % gpu gather
                            all_power_block{1,r_segment}{r_embedding,1}=gather(temp_power); % gpu gather
                            all_bregma_block{1,r_segment}{r_embedding,1}=temp_bregma;
                            all_loss_block{1,r_segment}{r_embedding,1}=temp_loss;
                            all_X_block{1,r_segment}{r_embedding,1}=temp_X;
                        end
                    end
                    
                    save(path_ICAspace_group_temp_block_temp,'all_EEG_block','all_Wchannel_ori_block','all_Wchannel_block','all_power_block','all_bregma_block','all_loss_block','all_X_block');
                    disp([path_ICAspace_group_temp_block_temp ' have been done.']);
                end
            end
            
            for r_block=1:nBlock
                
                all_EEG_block=[];
                all_Wchannel_ori_block=[];
                all_Wchannel_block=[];
                all_power_block=[];
                all_bregma_block=[];
                all_loss_block=[];
                all_X_block=[];
                path_ICAspace_group_temp_block_temp=[path_ICAspace_group_temp_block temp_name(1:end-4) '_block_' num2code(r_block,n_code) '.mat'];
                load(path_ICAspace_group_temp_block_temp);
                
                all_EEG{r_block,1}=all_EEG_block;
                all_Wchannel_ori{r_block,1}=all_Wchannel_ori_block;
                all_Wchannel{r_block,1}=all_Wchannel_block;
                all_power{r_block,1}=all_power_block;
                all_bregma{r_block,1}=all_bregma_block;
                all_loss{r_block,1}=all_loss_block;
                all_X{r_block,1}=all_X_block;
            end
            
            save(path_ICAspace_group_temp,'all_EEG','all_Wchannel_ori','all_Wchannel','all_power','all_bregma','all_loss','all_X');
            disp([path_ICAspace_group_temp ' have been done.']);
        end
        
        if figureShow~=0
            all_Wchannel=[];
            all_power=[];
            all_bregma=[];
            all_loss=[];
            all_X=[];
            load(path_ICAspace_group_temp)
            nBlock=size(all_Wchannel,1);
            
            for r_block=1:nBlock
                
                nSegment=size(all_Wchannel{r_block,1},2);
                for r_segment=1:nSegment
                    for r_embedding=1:nEmbedding
                        
                        temp_power=all_power{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_bregma=all_bregma{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_loss=all_loss{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_X=all_X{r_block,1}{1,r_segment}{r_embedding,1};
                        
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
                        
                        if figureShow==1 && r_embedding==nEmbedding
                            close all;
                        elseif figureShow==2 && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        elseif figureShow==3 && r_block==nBlock && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        elseif figureShow==4 && r_subject==nSubject && r_block==nBlock && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        elseif figureShow==5 && r_group==nGroup && r_subject==nSubject && r_block==nBlock && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        end
                    end
                end
            end
        end
    end
end

xCloud=linspace(-nRangeRate,nRangeRate,2*nCloud);
yCloud=linspace(-nRangeRate,nRangeRate,2*nCloud);
zCloud=linspace(-nRangeRate,0,nCloud);

xEdge=zeros(8,1);
yEdge=zeros(8,1);
zEdge=zeros(8,1);
vEdge=zeros(8,1);
rEdge=0;
for r1=[-nRangeRate,nRangeRate]
    for r2=[-nRangeRate,nRangeRate]
        for r3=[-nRangeRate,0]
            
            rEdge=rEdge+1;
            xEdge(rEdge,1)=r1;
            yEdge(rEdge,1)=r2;
            zEdge(rEdge,1)=r3;
        end
    end
end

for r_group=rGroup
    
    path_ICAspace_group=[path_ICAspace 'group_' num2code(r_group,n_code) '\'];
    D=dir([path_ICAspace_group '*.mat']);
    
    path_ICAcloud_group=[path_ICAcloud 'group_' num2code(r_group,n_code) '\'];
    if ~exist(path_ICAcloud_group,'dir')
        mkdir(path_ICAcloud_group)
    end
    
    nSubject=length(D);
    for r_subject=1:nSubject
        
        path_ICAspace_group_temp=[path_ICAspace_group D(r_subject).name];
        path_ICAcloud_group_temp=[path_ICAcloud_group D(r_subject).name];
        if exist(path_ICAcloud_group_temp,'file')
            disp([path_ICAcloud_group_temp ' exist, skip.']);
        else
            all_EEG=[];
            all_Wchannel_ori=[];
            all_Wchannel=[];
            all_power=[];
            all_bregma=[];
            all_loss=[];
            all_X=[];
            load(path_ICAspace_group_temp);
            
            all_power_cloud=[];
            all_bregma_cloud=[];
            all_loss_cloud=[];
            
            nBlock=size(all_EEG,1);
            for r_block=1:nBlock
                
                nSegment=size(all_EEG{r_block,1},2);
                for r_segment=1:nSegment
                    
                    disp(['ICAcloud: rBlock/nBlock: ' num2str(r_block) '/' num2str(nBlock) ', rSegment/nSegment: ' num2str(r_segment) '/' num2str(nSegment)]);
                    for r_embedding=1:nEmbedding
                        
                        temp_power=all_power{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_bregma=all_bregma{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_loss=all_loss{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_X=all_X{r_block,1}{1,r_segment}{r_embedding,1};
                        for r_cloud=1:3
                            if r_cloud==1
                                temp_cloud_input=temp_power;
                            elseif r_cloud==2
                                temp_cloud_input=temp_bregma;
                            elseif r_cloud==3
                                temp_cloud_input=temp_loss;
                            end
                            
                            temp_F=scatteredInterpolant([xEdge;temp_X(:,1)],[yEdge;temp_X(:,2)],[zEdge;temp_X(:,3)],[vEdge;temp_cloud_input],'linear');
                            temp_cloud_output=zeros(2*nCloud,2*nCloud,nCloud);
                            
                            r0_1=0;
                            for r1=xCloud
                                
                                r0_1=r0_1+1;
                                r0_2=0;
                                for r2=yCloud
                                    
                                    r0_2=r0_2+1;
                                    r0_3=0;
                                    for r3=zCloud
                                        
                                        r0_3=r0_3+1;
                                        
                                        temp_cloud_output(r0_1,r0_2,r0_3)=temp_F(r1,r2,r3);
                                    end
                                end
                            end
                            
                            temp_cloud_output(isnan(temp_cloud_output))=0;
                            
                            if r_cloud==1
                                temp_power_cloud=temp_cloud_output;
                            elseif r_cloud==2
                                temp_bregma_cloud=temp_cloud_output;
                            elseif r_cloud==3
                                temp_loss_cloud=temp_cloud_output;
                            end
                        end
                        
                        all_power_cloud{r_block,1}{1,r_segment}{r_embedding,1}=temp_power_cloud;
                        all_bregma_cloud{r_block,1}{1,r_segment}{r_embedding,1}=temp_bregma_cloud;
                        all_loss_cloud{r_block,1}{1,r_segment}{r_embedding,1}=temp_loss_cloud;
                    end
                end
            end
            
            save(path_ICAcloud_group_temp,'all_power_cloud','all_bregma_cloud','all_loss_cloud');
            disp([path_ICAcloud_group_temp ' have been done.']);
        end
        
        if figureShow_cloud~=0
            all_power_cloud=[];
            all_bregma_cloud=[];
            all_loss_cloud=[];
            load(path_ICAcloud_group_temp)
            nBlock=size(all_bregma_cloud,1);
            
            for r_block=1:nBlock
                
                nSegment=size(all_bregma_cloud{r_block,1},2);
                for r_segment=1:nSegment
                    for r_embedding=1:nEmbedding
                        
                        temp_power_cloud=all_power_cloud{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_bregma_cloud=all_bregma_cloud{r_block,1}{1,r_segment}{r_embedding,1};
                        temp_loss_cloud=all_loss_cloud{r_block,1}{1,r_segment}{r_embedding,1};
                        
                        figure
                        if figureSize{1,1}==1
                            set(gcf,'position',figureSize{2,1});
                        end
                        
                        scalp_outline;
                        
                        temp_colormap=jet(nColor);
                        for r1=1:nChannel
                            
                            hold on;
                            plot3(coordinate(r1,1),coordinate(r1,2),0,'o','MarkerSize',3,'MarkerFaceColor','k','MarkerEdgeColor','k');
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
                                    if temp_v>thrBregma
                                        plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',temp_colormap(rColor,:),'MarkerEdgeColor',temp_colormap(rColor,:));
                                    else
                                        grayRate=0.8;
                                        plot3(r1,r2,r3,'o','MarkerSize',markerSizeCloud,'MarkerFaceColor',grayRate*ones(1,3),'MarkerEdgeColor',grayRate*ones(1,3));
                                    end
                                end
                            end
                        end
                        
                        caxis(rangeBregma_cloud);
                        axis([-rRange rRange -rRange rRange]);
                        axis equal;
                        axis off;
                        view(Rotation,Tumbling);
                        
                        if figureShow_cloud==1 && r_embedding==nEmbedding
                            close all;
                        elseif figureShow_cloud==2 && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        elseif figureShow_cloud==3 && r_block==nBlock && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        elseif figureShow_cloud==4 && r_subject==nSubject && r_block==nBlock && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        elseif figureShow_cloud==5 && r_group==nGroup && r_subject==nSubject && r_block==nBlock && r_segment==nSegment && r_embedding==nEmbedding
                            close all;
                        end
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
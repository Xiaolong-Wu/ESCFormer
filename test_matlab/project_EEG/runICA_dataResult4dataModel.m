tic
% close all;clear all;clc;
%%

% 0, data_2024_3D
% 1, data_2024_Competition
% 2, data_2024_DoC sin1200
% 3, data_2024_DoC dogbark
% 4, data_2024_DoC xiaoming
dataNo=1;
n_code=3;

% modelNo(1)
% 0, modelSpace
% 1, modelCloud

% modelNo(2)
% 0, EEGNet
% 1, CNN-LSTM
% 2, CNN-Transformer
% 3, TimesNet
% 4, TimesBlock-LSTM
% 5, TimesBlock-Transformer
% modelNo=[0,0];

% modelEpoch=64;
modelBatchSize=32;

% 1, lossAll
% 2, lossGather
% 3, confusionMatrixBar
% 4, confusionMatrix
% 5, individual
figureNo=[2 4];

typeIndividual=0; % 0, 1
typeIndicator=1; % macro-: 1, precision; 2, recall; 3~, F~ score

typeLoss=1; % 1, mean; 2, median
typeLossRange=2; % 1, std; 2, extrem

% macro-average
Fscore=[1,2];

lineWidth=2;
limLoss=[0.2,1.4];

faceColor='r';
faceAlpha=0.3;

limBar=[0,1];

figureSize1={0;[300,300,300,150]};
figureSize2={0;[300,300,600,250]};
figureSize3={1;[300,300,600,250]};
%%

path_temp=[pwd '\'];
path_python='D:\test_pycharm\project_eegICAcloud\';

path_ICA={
    'data_2024_3D\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_Competition\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\'
    };
path_ICAspace={
    'data_2024_3D\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)\';
    'data_2024_Competition\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)_1_sin1200\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)_2_dogbark\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)_3_xiaoming\'
    };
path_ICAcloud={
    'data_2024_3D\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_Competition\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)_1_sin1200\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)_2_dogbark\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)_3_xiaoming\'
    };

groupAll={
    [];
    [1:4];
    [13,22];
    [16,25];
    [18,27]
    };

path_toolbox='D:\test_matlab\toolbox\eegGA\';
addpath(genpath(path_toolbox));
%%

pathICAdata=path_ICA{dataNo+1,1};
if modelNo(1)==0
    pathICAmodel=path_ICAspace{dataNo+1,1};
elseif modelNo(1)==1
    pathICAmodel=path_ICAcloud{dataNo+1,1};
end

path_data_pythonTest=[path_python 'dataset_test\' pathICAdata];
path_data_pythonModel=[path_python 'dataset_model\' pathICAmodel];

modelName=['modelNo(' num2str(modelNo(2)) ')epoch(' num2str(modelEpoch) ')batch(' num2str(modelBatchSize) ')'];
path_target=[path_data_pythonModel modelName '_target.txt'];
path_loss=[path_data_pythonModel modelName '_loss.txt'];
path_test=[path_data_pythonModel modelName '_test.txt'];

temp_target=load(path_target);
temp_loss=load(path_loss);
temp_test=load(path_test);

group_all=groupAll{dataNo+1,1};
nGroup=length(group_all);

nTemp_loss=cell(nGroup,1);
for r1=1:nGroup
    
    path_data_pythonTest_group=[path_data_pythonTest 'group_' num2code(group_all(r1),n_code) '\'];
    D=dir([path_data_pythonTest_group '*.mat']);
    nSubject=length(D);
    for r2=1:nSubject
        
        path_data_pythonTest_group_temp=[path_data_pythonTest_group D(r2).name];
        
        if modelNo(1)==0
            all_EEG=[];
            load(path_data_pythonTest_group_temp);
            
            nTemp_loss{r1,1}(r2,1)=size(all_EEG,1);
        elseif modelNo(1)==1
            all_power_cloud=[];
            all_bregma_cloud=[];
            all_loss_cloud=[];
            load(path_data_pythonTest_group_temp);
            
            nTemp_loss{r1,1}(r2,1)=size(all_bregma_cloud,1);
        end
    end
end
%%

epochAll=size(temp_loss,1);
epochInterval=epochAll/modelEpoch;

xEpoch=([1:epochAll])./epochInterval;

if ismember(1,figureNo)
    figure
    if figureSize1{1,1}==1
        set(gcf,'position',figureSize1{2,1});
    end
    plot(xEpoch,temp_loss,'k','LineWidth',lineWidth);
    xlim([0,modelEpoch]);
    ylim(limLoss);
end
%%

temp_loss_m=zeros(modelEpoch,1);
temp_loss_extremMin=zeros(modelEpoch,1);
temp_loss_extremMax=zeros(modelEpoch,1);
for r=1:modelEpoch
    
    first=(r-1)*epochInterval+1;
    last=r*epochInterval;
    
    temp_loss_gather=temp_loss(first:last,1);
    if typeLoss==1
        temp_loss_m(r,1)=mean(temp_loss_gather);
    elseif typeLoss==2
        temp_loss_m(r,1)=median(temp_loss_gather);
    end
    
    if typeLossRange==1
        temp_loss_extremMin(r,1)=-std(temp_loss_gather);
        temp_loss_extremMax(r,1)=-temp_loss_extremMin(r,1);
    elseif typeLossRange==2
        temp_loss_extremMin(r,1)=min(temp_loss_gather)-temp_loss_m(r,1);
        temp_loss_extremMax(r,1)=max(temp_loss_gather)-temp_loss_m(r,1);
    end
end

if ismember(2,figureNo)
    figure
    if figureSize1{1,1}==1
        set(gcf,'position',figureSize1{2,1});
    end
    area(1:modelEpoch,temp_loss_m+temp_loss_extremMax,'FaceColor',faceColor,'EdgeColor','w','FaceAlpha',faceAlpha);
    hold on;
    area(1:modelEpoch,temp_loss_m+temp_loss_extremMin,'FaceColor','w','EdgeColor','w');
    hold on;
    plot(1:modelEpoch,temp_loss_m,'k','LineWidth',lineWidth);
    xlim([0,modelEpoch]);
    ylim(limLoss);
end
%%

[testAll,classAll]=size(temp_target);

confusionMatrix=zeros(classAll,classAll);
for r=1:testAll
    
    [~,a]=max(temp_target(r,:));
    [~,b]=max(temp_test(r,:));
    
    confusionMatrix(a,b)=confusionMatrix(a,b)+1;
end

confusionMatrix_rate=confusionMatrix;
for r=1:classAll
    
    confusionMatrix_rate(r,:)=confusionMatrix_rate(r,:)/sum(confusionMatrix_rate(r,:));
end

figure
if figureSize2{1,1}==1
    set(gcf,'position',figureSize2{2,1});
end
caxis([0,1]);
colormapTemp=colormap(bone);
axis off;
colorbar('FontWeight','bold');
if ~ismember(3,figureNo)
    close;
end

nColormapTemp=size(colormapTemp,1);

if ismember(4,figureNo)
    figure
    if figureSize2{1,1}==1
        set(gcf,'position',figureSize2{2,1});
    end
    imagesc(confusionMatrix_rate);
    
    for a=1:classAll
        for b=1:classAll
            
            temp_accRate=confusionMatrix_rate(a,b);
            temp_accSample=confusionMatrix(a,b);
            if temp_accRate>0.5
                rColormapTemp=1;
            else
                rColormapTemp=nColormapTemp;
            end
            text(b,a,num2str(temp_accSample),'Color',colormapTemp(rColormapTemp,:),'HorizontalAlignment','center','FontWeight','bold');
        end
    end
    caxis([0,1]);
    colormap(bone);
    set(gca,'XAxisLocation','top');
    axis equal;
    axis off;
end
%%

nFscore=size(Fscore,2);

macro_precision_all=zeros(1,classAll);
macro_recall_all=zeros(1,classAll);
macro_F_all=zeros(nFscore,classAll);
for r1=1:classAll
    
    macro_precision_temp=confusionMatrix(r1,r1)/sum(confusionMatrix(:,r1));
    macro_recall_temp=confusionMatrix(r1,r1)/sum(confusionMatrix(r1,:));
    
    macro_precision_all(1,r1)=macro_precision_temp;
    macro_recall_all(1,r1)=macro_recall_temp;
    for r2=1:nFscore
        
        Fscore_temp=Fscore(1,r2);
        macro_F_temp=(Fscore_temp^2+1)*macro_precision_temp*macro_recall_temp/((Fscore_temp^2)*(macro_precision_temp+macro_recall_temp));
        macro_F_all(r2,r1)=macro_F_temp;
    end
end
macro_precision=mean(macro_precision_all);
macro_recall=mean(macro_recall_all);
macro_F=mean(macro_F_all,2);

disp(' ');
disp(['macro precision: ' num2str(macro_precision)]);
disp(['macro recall: ' num2str(macro_recall)]);
for r=1:nFscore
    disp(['macro F' num2str(Fscore(1,r)) ' score: ' num2str(macro_F(r,1))]);
end
%%

if typeIndividual==1 || ismember(5,figureNo)
    
    all_confusionMatrix=cell(nGroup,1);
    nTemp=0;
    for r1=1:nGroup
        for r2=1:size(nTemp_loss{r1,1},1)
            
            first=nTemp+1;
            
            nTemp=nTemp+nTemp_loss{r1,1}(r2,1);
            last=nTemp;
            
            temp_temp_target=temp_target(first:last,:);
            temp_temp_test=temp_test(first:last,:);
            
            temp_testAll=size(temp_temp_target,1);
            
            temp_confusionMatrix=zeros(classAll,classAll);
            for r3=1:temp_testAll
                
                [~,a]=max(temp_temp_target(r3,:));
                [~,b]=max(temp_temp_test(r3,:));
                
                temp_confusionMatrix(a,b)=temp_confusionMatrix(a,b)+1;
            end
            
            all_confusionMatrix{r1,1}{r2,1}=temp_confusionMatrix;
        end
    end
    
    individual_confusionMatrix=[];
    if dataNo==1
        for r1=1:nGroup
            
            if r1==1
                individual_confusionMatrix=all_confusionMatrix{r1,1}';
            else
                for r2=1:size(nTemp_loss{r1,1},1)
                    
                    individual_confusionMatrix{1,r2}=individual_confusionMatrix{1,r2}+all_confusionMatrix{r1,1}{r2,1};
                end
            end
        end
    else
        r0=0;
        for r1=1:nGroup
            for r2=1:size(nTemp_loss{r1,1},1)
                
                r0=r0+1;
                individual_confusionMatrix{1,r0}=all_confusionMatrix{r1,1}{r2,1};
            end
        end
    end
    
    nIndividual=size(individual_confusionMatrix,2);
    individual_confusionMatrix_rate=individual_confusionMatrix;
    for r1=1:nIndividual
        for r2=1:classAll
            
            individual_confusionMatrix_rate{1,r1}(r2,:)=individual_confusionMatrix_rate{1,r1}(r2,:)/sum(individual_confusionMatrix_rate{1,r1}(r2,:));
        end
    end
    
    if ismember(4,figureNo)
        for r=1:nIndividual
            
            temp_confusionMatrix=individual_confusionMatrix{1,r};
            temp_confusionMatrix_rate=individual_confusionMatrix_rate{1,r};
            figure
            if figureSize2{1,1}==1
                set(gcf,'position',figureSize2{2,1});
            end
            imagesc(temp_confusionMatrix_rate);
            
            for a=1:classAll
                for b=1:classAll
                    
                    temp_accRate=temp_confusionMatrix_rate(a,b);
                    temp_accSample=temp_confusionMatrix(a,b);
                    if temp_accRate>0.5
                        rColormapTemp=1;
                    else
                        rColormapTemp=nColormapTemp;
                    end
                    text(b,a,num2str(temp_accSample),'Color',colormapTemp(rColormapTemp,:),'HorizontalAlignment','center','FontWeight','bold');
                end
            end
            caxis([0,1]);
            colormap(bone);
            set(gca,'XAxisLocation','top');
            axis equal;
            axis off;
        end
    end
    %%
    
    indicator_all=zeros(1,nIndividual);
    for r1=1:nIndividual
        
        temp_confusionMatrix=individual_confusionMatrix{1,r1};
        
        macro_precision_all=zeros(1,classAll);
        macro_recall_all=zeros(1,classAll);
        macro_F_all=zeros(nFscore,classAll);
        for r2=1:classAll
            
            macro_precision_temp=temp_confusionMatrix(r2,r2)/sum(temp_confusionMatrix(:,r2));
            macro_recall_temp=temp_confusionMatrix(r2,r2)/sum(temp_confusionMatrix(r2,:));
            
            macro_precision_all(1,r2)=macro_precision_temp;
            macro_recall_all(1,r2)=macro_recall_temp;
            for r3=1:nFscore
                
                Fscore_temp=Fscore(1,r3);
                macro_F_temp=(Fscore_temp^2+1)*macro_precision_temp*macro_recall_temp/((Fscore_temp^2)*(macro_precision_temp+macro_recall_temp));
                macro_F_all(r3,r2)=macro_F_temp;
            end
        end
        macro_precision=mean(macro_precision_all);
        macro_recall=mean(macro_recall_all);
        macro_F=mean(macro_F_all,2);
        
        disp(' ');
        disp(['macro precision: ' num2str(macro_precision)]);
        disp(['macro recall: ' num2str(macro_recall)]);
        for r2=1:nFscore
            disp(['macro F' num2str(Fscore(1,r2)) ' score: ' num2str(macro_F(r2,1))]);
        end
        
        if typeIndicator==1
            indicator_all(1,r1)=macro_precision;
        elseif typeIndicator==2
            indicator_all(1,r1)=macro_recall;
        elseif typeIndicator==3
            indicator_all(1,r1)=macro_F(1,1);
        end
    end
    
    if ismember(5,figureNo)
        figure
        if figureSize3{1,1}==1
            set(gcf,'position',figureSize2{2,1});
        end
        bar(indicator_all);
        ylim(limBar);
    end
end
%%

rmpath(genpath(path_toolbox));
toc
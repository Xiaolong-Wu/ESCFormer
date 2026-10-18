tic
close all;clear all;clc;
%%

nChannel=22;

rGroup=2;
rSubject=0; % 0 mean all;
rChannel=13;
filterBand=[2,8];

fs=250;
T=0.8;
olr=0;
T_window='';

Band=[2,8];
Scale_v=[0,12;300,400;0,100;0,1];
Scale_t=[0,13;300,400;0,100;0,1];

n_code=3;
vLim=[-20,20];

ERP_show=[4]; % 1, ERP; 2, ERP shift; 3, ERP mean std; 4, ERP median prctile
timeStamp_show=1;
lineWidth=2;

path_data=[pwd '\'];
path_toolbox='D:\test_matlab\toolbox\eegGA\function\';
%%

addpath(genpath(path_toolbox));
path_block=([path_data num2str(T) 's(' num2str(olr) ')' T_window '\block\group_' num2code(rGroup,n_code) '\']);

file_name_v=['timeFeature' '(' num2str(Band(1,1)) ',' num2str(Band(1,2)) ')Hz(' num2str(Scale_v(1,2)) ')'];
file_name_t=['timeFeature' '(' num2str(Band(1,1)) ',' num2str(Band(1,2)) ')Hz(' num2str(Scale_t(1,2)) ')'];

file_name_temp_v=['tBehind(' num2str(Scale_v(3,2)) ')tSpan(' num2str(Scale_v(2,1)) ',' num2str(Scale_v(2,2)) ')isNan(' num2str(Scale_v(4,2)) ')'];
file_name_temp_t=['tBehind(' num2str(Scale_t(3,2)) ')tSpan(' num2str(Scale_t(2,1)) ',' num2str(Scale_t(2,2)) ')isNan(' num2str(Scale_t(4,2)) ')'];

path_ERP_v=([path_data num2str(T) 's(' num2str(olr) ')' T_window '\' file_name_v file_name_temp_v '\group_' num2code(rGroup,n_code) '\']);
path_ERP_t=([path_data num2str(T) 's(' num2str(olr) ')' T_window '\' file_name_t file_name_temp_t '\group_' num2code(rGroup,n_code) '\']);
%%

D=dir([path_block '*.mat']);
load(['scalp(' num2str(nChannel) ').mat']);

rAll=0;
rERP=0;
all_ERP=cell(1,2);

if rSubject==0
    rSubject=1:length(D);
end
for r1=rSubject
    
    block=[];
    load([path_block D(r1).name]);
    temp_block=block;
    
    sample=[];
    load([path_ERP_v D(r1).name]);
    temp_ERP_v=sample;
    
    sample=[];
    load([path_ERP_t D(r1).name]);
    temp_ERP_t=sample;
    
    nBlock=size(temp_block,1);
    for r2=1:nBlock
        
        nSegment=size(temp_block{r2,1},2);
        for r3=1:nSegment
            
            rAll=rAll+1;
            p_t=temp_ERP_t{r2,1}{1,r3}{1,1}(1,rChannel);
            if ~isnan(p_t)
                rERP=rERP+1;
                
                temp_block_s=temp_block{r2,1}{1,r3}(rChannel,:);
                
                [~,s_PS]=wave_PS(temp_block_s,fs,filterBand,[0 0],'flat');
                temp_block_ERP=s_PS{1,1}(1,:);
                
                nBehind=ceil(Scale_t(3,2)*fs/1000);
                temp_block_ERP=temp_block_ERP-mean(temp_block_ERP(1:nBehind));
                
                all_ERP{1,1}(rERP,:)=temp_block_ERP;
                all_ERP{1,2}(rERP,1)=p_t;
            end
        end
    end
end

disp(' ');
disp(['ch.' coordinate_name{rChannel,1}]);
disp(['all: ' num2str(rAll)]);
disp(['non: ' num2str(rERP)]);

[nERP,nT]=size(all_ERP{1,1});
t=(1000*(0:nT-1)/fs)-Scale_t(3,2);

if ismember(1,ERP_show)
    figure
    for r=1:nERP
        
        plot(t,all_ERP{1,1}(r,:));
        hold on;
        
        if timeStamp_show==1
            [~,ind_t]=min(abs(t-all_ERP{1,2}(r,1)));
            plot([t(ind_t),t(ind_t)],[0 all_ERP{1,1}(r,ind_t)],'k');
            hold on;
        end
    end
    
    plot([-Scale_t(3,2),1000*T-Scale_t(3,2)],[0,0],'k','LineWidth',lineWidth);
    hold on;
    plot([0,0],vLim,'k','LineWidth',lineWidth);
    xlim([-Scale_t(3,2),1000*T-Scale_t(3,2)]);
    ylim(vLim);
    
    title(['ch.' coordinate_name{rChannel,1}]);
end
%%

all_ERP_shift=all_ERP;
for r=1:nERP
    
    temp_ERP=[zeros(1,nT),all_ERP{1,1}(r,:),zeros(1,nT)];
    
    [~,ind_t_1]=min(abs(t-all_ERP{1,2}(r,1)));
    [~,ind_t_2]=min(abs(t-mean(Scale_t(2,:))));
    
    first=nT+ind_t_1-ind_t_2+1;
    last=first+nT-1;
    
    all_ERP_shift{1,1}(r,:)=temp_ERP(first:last);
    all_ERP_shift{1,2}(r,1)=mean(Scale_t(2,:));
end

if ismember(2,ERP_show)
    figure
    for r=1:nERP
        
        plot(t,all_ERP_shift{1,1}(r,:));
        hold on;
        
        if timeStamp_show==1
            [~,ind_t]=min(abs(t-all_ERP_shift{1,2}(r,1)));
            plot([t(ind_t),t(ind_t)],[0 all_ERP_shift{1,1}(r,ind_t)],'k');
            hold on;
        end
    end
    
    plot([-Scale_t(3,2),1000*T-Scale_t(3,2)],[0,0],'k','LineWidth',lineWidth);
    hold on;
    plot([0,0],vLim,'k','LineWidth',lineWidth);
    xlim([-Scale_t(3,2),1000*T-Scale_t(3,2)]);
    ylim(vLim);
    
    title(['ch.' coordinate_name{rChannel,1}]);
end
%%

all_ERP_statistics=all_ERP;
% all_ERP_statistics=all_ERP_shift;
%%

if ismember(3,ERP_show)
    figure
    
    mean_ERP=mean(all_ERP_statistics{1,1},1);
    std_ERP=std(all_ERP_statistics{1,1},0,1);
    
    plot(t,mean_ERP,'LineWidth',lineWidth);
    hold on;
    plot(t,mean_ERP+std_ERP);
    hold on;
    plot(t,mean_ERP-std_ERP);
    
    plot([-Scale_t(3,2),1000*T-Scale_t(3,2)],[0,0],'k','LineWidth',lineWidth);
    hold on;
    plot([0,0],vLim,'k','LineWidth',lineWidth);
    xlim([-Scale_t(3,2),1000*T-Scale_t(3,2)]);
    ylim(vLim);
    
    title(['ch.' coordinate_name{rChannel,1}]);
end
%%

if ismember(4,ERP_show)
    figure
    
    median_ERP=median(all_ERP_statistics{1,1},1);
    prctile_ERP1=prctile(all_ERP_statistics{1,1},75,1);
    prctile_ERP2=prctile(all_ERP_statistics{1,1},25,1);
    
    plot(t,median_ERP,'LineWidth',lineWidth);
    hold on;
    plot(t,prctile_ERP1);
    hold on;
    plot(t,prctile_ERP2);
    
    plot([-Scale_t(3,2),1000*T-Scale_t(3,2)],[0,0],'k','LineWidth',lineWidth);
    hold on;
    plot([0,0],vLim,'k','LineWidth',lineWidth);
    xlim([-Scale_t(3,2),1000*T-Scale_t(3,2)]);
    ylim(vLim);
    
    title(['ch.' coordinate_name{rChannel,1}]);
end
%%

rmpath(genpath(path_toolbox));
toc
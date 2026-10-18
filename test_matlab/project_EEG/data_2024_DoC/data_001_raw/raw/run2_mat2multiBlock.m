tic
close all;clear all;clc;
%%

typeT=1; % 1 2

if typeT==1
    % unit, msec
    T=[-100,700];
    filterBand={
        0,[45,55];
        1,[30,60]; % [1,4] [4,8] [8,13] [13,30] [30,60] [1,300]
        };
elseif typeT==2
    % unit, msec
    T=[300,400];
    filterBand={
        1,[1,5];
        };
end

fs=1000;
n_code=3;

path_suffix=['(' num2str(T(1,1)) ',' num2str(T(1,2)) ')'];
path_old='D:\test_matlab\project_EEG\data_2024_DoC\data_001_raw\raw\EEG(mat)\';
path_new=['D:\test_matlab\project_EEG\data_2024_DoC\data_001_raw\temp_group' path_suffix '\'];

info_trial={
    1,'Бебл10','EC';
    2,'еібл10','EO';
    3,'MMN','MMN';
    4,'P3A','P3A';
    5,'P300','P300'
    };

info_voice={
    1,'none';
    2,'sin800';
    3,'sin1000';
    4,'sin1200';
    5,'dog';
    6,'xiaoming'
    };

info_group={
    [1];
    [1];
    [2,4];
    [2,4,5];
    [3,6];
    };

path_toolbox='D:\test_matlab\toolbox\eegGA\';
%%

addpath(genpath(path_toolbox));
%%

if ~exist(path_new,'dir')
    mkdir(path_new);
end

r0_1=0;
nameGroup=[];
nTrial=size(info_trial,1);
for r1=1:nTrial
    
    nStim=length(info_group{r1,1});
    for r2=1:nStim
        
        r0_1=r0_1+1;
        r_voice=info_group{r1,1}(r2);
        nameGroup{r0_1,1}=['group_' num2code(r0_1,n_code) ': ' info_trial{r1,3} ' (voice: ' info_voice{r_voice,2} ')'];
        path_oldGroup=[path_old info_trial{r1,3} '\'];
        path_newGroup=[path_new 'group_' num2code(r0_1,n_code) '\'];
        if ~exist(path_newGroup,'dir')
            mkdir(path_newGroup);
        end
        D=dir([path_oldGroup '*.mat']);
        for r3=1:length(D)
            
            path_oldGroup_temp=[path_oldGroup D(r3).name];
            path_newGroup_temp=[path_newGroup D(r3).name];
            if exist(path_newGroup_temp,'file')
                disp([path_newGroup_temp ' exist, skip.']);
            else
                EEG=[];
                voice_stim=[];
                load(path_oldGroup_temp);
                
                [nChannel,nT]=size(EEG{1,1});
                for r4=1:nChannel
                    
                    s_filter=double(EEG{1,1}(r4,:));
                    for r5=1:size(filterBand,1)
                        
                        if filterBand{r5,1}==0
                            s_filter_1=waveFilter(s_filter,fs,[0,filterBand{r5,2}(1,1)]);
                            s_filter_2=waveFilter(s_filter,fs,[filterBand{r5,2}(1,2),inf]);
                            s_filter=s_filter_1+s_filter_2;
                        elseif filterBand{r5,1}==1
                            s_filter=waveFilter(s_filter,fs,filterBand{r5,2});
                        end
                    end
                    
                    EEG{1,1}(r4,:)=s_filter;
                end
                
                if r_voice~=1
                    EEG_temp=EEG;
                    EEG=[];
                    
                    r0_2=0;
                    for r4=1:nT
                        
                        if voice_stim{1,1}(1,r4)==r2
                            first=r4+T(1,1)*fs/1000;
                            last=r4-1+T(1,2)*fs/1000;
                            
                            if first>0 && last<=nT
                                r0_2=r0_2+1;
                                EEG{r0_2,1}=EEG_temp{1,1}(:,first:last);
                            end
                        end
                    end
                end
                save(path_newGroup_temp,'EEG');
                disp([path_newGroup_temp ' have been done.']);
            end
        end
    end
end
%%

disp(nameGroup);
rmpath(genpath(path_toolbox));
toc
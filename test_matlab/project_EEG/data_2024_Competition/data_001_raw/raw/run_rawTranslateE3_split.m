tic
close all;clear all;clc;
%%

path_dataOld='D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\dataSet_translateE2\';
path_dataNew='D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\dataSet_translateE3(-100,700)\';
path_toolbox='D:\test_matlab\toolbox\eegGA\';

subNo=[1 2 3 4 5 6 7 8 9];
runNo={'T'}; % {'E';'T'}
fs=250;
nClass=4; % save(path_dataNew_temp,'EEG_class1','EEG_class2','EEG_class3','EEG_class4');

Select=0;
energyDecreasingBand=[8,45];

T_dimension=0.8; % the total time dimension used for identification, unit: sec
centerRate=1/8; % the timestamp location rate of T_demension
%%

if ~exist(path_dataNew,'dir')
    mkdir(path_dataNew);
end
%%

addpath(genpath(path_toolbox));

nSubNo=length(subNo);
nRunNo=length(runNo);
for r1=1:nSubNo
    
    cSubNo=num2code(subNo(r1),2);
    for r2=1:nRunNo
        
        path_dataOld_temp=[path_dataOld 'A' cSubNo runNo{r2,1} '.mat'];
        path_dataNew_temp=[path_dataNew 'A' cSubNo runNo{r2,1} '.mat'];
        
        if exist(path_dataNew_temp,'file')
            disp([path_dataNew_temp ' exist, skip.']);
        else
            load(path_dataOld_temp);
            
            n_all=size(EEG,2);
            n_T_dimension=ceil(T_dimension*fs);
            n_T_dimensionL=ceil(T_dimension*fs*centerRate);
            n_T_dimensionR=n_T_dimension-n_T_dimensionL;
            n_event=size(Event);
            
            for r3=1:nClass
                
                class_temp=num2str(r3);
                eval(['r_class',class_temp,'=0;']);
                eval(['EEG_class',class_temp,'=[];']);
            end
            
            for r3=1:n_event
                
                first=Event(r3,1)+1-n_T_dimensionL;
                last=first-1+n_T_dimension;
                
                if first<1
                    continue;
                elseif last>n_all
                    break;
                end
                
                EEG_temp=EEG(:,first:last);
                
                nChannel=size(EEG_temp,1);
                ind_all=zeros(nChannel,1);
                for r4=1:nChannel
                    
                    ind_all(r4,1)=selectRule(EEG_temp(r4,:),fs,energyDecreasingBand);
                end
                
                if Select==0 || nChannel==sum(ind_all)
                    class_temp=num2str(Event(r3,2));
                    eval(['r_class',class_temp,'=r_class',class_temp,'+1;']);
                    eval(['EEG_class',class_temp,'{r_class',class_temp,',1}=EEG_temp;']);
                end
            end
            
            save(path_dataNew_temp,'EEG_class1','EEG_class2','EEG_class3','EEG_class4');
            disp([path_dataNew_temp ' have been done.']);
        end
    end
end
rmpath(genpath(path_toolbox));
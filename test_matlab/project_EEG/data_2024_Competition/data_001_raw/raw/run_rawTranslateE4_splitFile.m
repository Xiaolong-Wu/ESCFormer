tic
close all;clear all;clc;
%%

path_dataOld='D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\dataSet_translateE3(-100,700)\';
path_dataNew='D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\dataSet_translateE4(-100,700)\';
path_toolbox='D:\test_matlab\toolbox\eegGA\';

subNo=[1 2 3 4 5 6 7 8 9];
runNo={'T'};
runGroups={'group_001';'group_002';'group_003';'group_004'}; % 'EEG_class1';'EEG_class2';'EEG_class3';'EEG_class4'
%%

if ~exist(path_dataNew,'dir')
    mkdir(path_dataNew);
end
%%

addpath(genpath(path_toolbox));

nSubNo=length(subNo);
nRunNo=length(runNo);
nGroups=length(runGroups);
for r1=1:nSubNo
    
    cSubNo=num2code(subNo(r1),2);
    for r2=1:nRunNo
        
        path_dataOld_temp=[path_dataOld 'A' cSubNo runNo{r2,1} '.mat'];
        for r3=1:nGroups
            
            path_dataNewGroupMI=[path_dataNew runGroups{r3,1} '\'];
            if ~exist(path_dataNewGroupMI,'dir')
                mkdir(path_dataNewGroupMI);
            end
            
            path_dataNewGroupMI_temp=[path_dataNewGroupMI 'A' cSubNo runNo{r2,1} '.mat'];
            if exist(path_dataNewGroupMI_temp,'file')
                disp([path_dataNewGroupMI_temp ' exist, skip.']);
            else
                load(path_dataOld_temp);
                
                class_temp=num2str(r3);
                eval(['EEG=EEG_class',class_temp,';']);
                
                save(path_dataNewGroupMI_temp,'EEG');
                disp([path_dataNewGroupMI_temp ' have been done.']);
            end
        end
    end
end
rmpath(genpath(path_toolbox));
toc
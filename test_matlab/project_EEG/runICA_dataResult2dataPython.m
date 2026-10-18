tic
close all;clear all;clc;
%%

testRate=-0.2;

% 1, data_2024_3D
% 2, data_2024_Competition
% 3, data_2024_DoC
dataNo=3;
n_code=3;
%%

path_temp=[pwd '\'];
path_python='D:\test_pycharm\project_eegICAcloud\';

path_ICAspace={
    'data_2024_3D\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)\';
    'data_2024_Competition\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAspace_Norm1(1)\'
    };
path_ICAcloud={
    'data_2024_3D\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_Competition\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\';
    'data_2024_DoC\data_003_ICA(-100,700)\0.8s(0)\ICAcloud_Norm1(1)\'
    };

groupAll={
    [];
    [1:4];
    [4,7,9,13,16,18,22,25,27]
    };

path_toolbox='D:\test_matlab\toolbox\eegGA\';
addpath(genpath(path_toolbox));
%%

for r_path=1:2
    
    if r_path==1
        pathICA=path_ICAspace{dataNo,1};
    elseif r_path==2
        pathICA=path_ICAcloud{dataNo,1};
    end
    
    path_data=[path_temp pathICA];
    path_data_pythonTrain=[path_python 'dataset_train\' pathICA];
    path_data_pythonTest=[path_python 'dataset_test\' pathICA];
    
    if ~exist(path_data_pythonTrain,'dir')
        mkdir(path_data_pythonTrain);
    end
    if ~exist(path_data_pythonTest,'dir')
        mkdir(path_data_pythonTest);
    end
    
    group_all=groupAll{dataNo,1};
    nGroup=length(group_all);
    for r1=1:nGroup
        
        path_data_group=[path_data 'group_' num2code(group_all(r1),n_code) '\'];
        
        path_data_pythonTrain_group=[path_data_pythonTrain 'group_' num2code(group_all(r1),n_code) '\'];
        path_data_pythonTest_group=[path_data_pythonTest 'group_' num2code(group_all(r1),n_code) '\'];
        
        if ~exist(path_data_pythonTrain_group,'dir')
            mkdir(path_data_pythonTrain_group);
        end
        if ~exist(path_data_pythonTest_group,'dir')
            mkdir(path_data_pythonTest_group);
        end
        
        D=dir([path_data_group '*.mat']);
        nSubject=length(D);
        for r2=1:nSubject
            
            path_data_group_temp=[path_data_group D(r2).name];
            
            path_data_pythonTrain_group_temp=[path_data_pythonTrain_group D(r2).name];
            path_data_pythonTest_group_temp=[path_data_pythonTest_group D(r2).name];
            
            if exist(path_data_pythonTrain_group_temp,'file') && exist(path_data_pythonTest_group_temp,'file')
                disp([path_data_pythonTrain_group_temp ' exist, skip.']);
                disp([path_data_pythonTest_group_temp ' exist, skip.']);
            else
                if r_path==1
                    all_EEG=[];
                    load(path_data_group_temp);
                    
                    all_data{1,1}=all_EEG;
                elseif r_path==2
                    all_power_cloud=[];
                    all_bregma_cloud=[];
                    all_loss_cloud=[];
                    load(path_data_group_temp);
                    
                    all_data{1,1}=all_power_cloud;
                    all_data{2,1}=all_bregma_cloud;
                    all_data{3,1}=all_loss_cloud;
                end
                
                temp_nSample=size(all_data{1,1});
                
                rTrain=[];
                rTest=[];
                rMod=1;
                for r3=1:temp_nSample
                    
                    if abs(r3*testRate)>=rMod
                        rMod=rMod+1;
                        rTest=[rTest;r3];
                    else
                        rTrain=[rTrain;r3];
                    end
                end
                
                if testRate<0
                    rTrain=[1:temp_nSample]';
                end
                
                for r3=1:size(all_data,1)
                    
                    train_data{r3,1}=all_data{r3,1}(rTrain,1);
                    test_data{r3,1}=all_data{r3,1}(rTest,1);
                end
                
                if r_path==1
                    all_EEG=train_data{1,1};
                    save(path_data_pythonTrain_group_temp,'all_EEG');
                    
                    all_EEG=test_data{1,1};
                    save(path_data_pythonTest_group_temp,'all_EEG');
                elseif r_path==2
                    all_power_cloud=train_data{1,1};
                    all_bregma_cloud=train_data{2,1};
                    all_loss_cloud=train_data{3,1};
                    save(path_data_pythonTrain_group_temp,'all_power_cloud','all_bregma_cloud','all_loss_cloud');
                    
                    all_power_cloud=test_data{1,1};
                    all_bregma_cloud=test_data{2,1};
                    all_loss_cloud=test_data{3,1};
                    save(path_data_pythonTest_group_temp,'all_power_cloud','all_bregma_cloud','all_loss_cloud');
                end
                
                disp([path_data_pythonTrain_group_temp ' have been done.']);
                disp([path_data_pythonTest_group_temp ' have been done.']);
            end
        end
    end
end
%%

rmpath(genpath(path_toolbox));
toc
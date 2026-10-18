%%

disp(' ');disp('setup: basic info');disp('<==============================>');
%% basic

path_data='D:\test_matlab\project_EEG\data_2024_DoC\data_002_T(-100,700)\';

n_code=3; % number of coding digit
n_group=27; % number of groups;
%% EEG info

% fs, sample frequency, unit: Hz (please ensure fs not more than 1000 by resample)

% channelNo, delect channel when its No.==0: y=channelNo(1,x)
% x is the row No. in EEG data
% y is the corresponding channel No. (location) in eegGA
% eg,
% [1:30]
% [1 2 5 3 25 4 6 9 7 26 8 10 13 11 27 12 14 17 15 28 16 18 21 19 29 20 22 23 30 24 0 0 0]

fs=1000; % 500, 1000
channelNo=[53,1,2,3,4,5,6,54,7,8,9,10,11,12,13,14,55,15,16,17,18,19,20,21,22,56,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,57,39,40,41,42,43,44,58,45,46,47,48,49,50,59,51,52,0,0,0,0,0];
%%

path_eegGA=[pwd '\'];
path_toolbox={
    [path_eegGA '\function\'];
    [path_eegGA '\toolbox\'];
    };

meth_all={
    'bandPower';
    'morphology';
    'correlation';
    'morphCorrelation';
    'mscohere';
    'morphMscohere';
    'PLV';
    'morphPLV';
    'GC';
    'morphGC';
    'CFC_bandPowerRate';
    'CFC_morphologyRate';
    'CFC_AAC';
    'CFC_morphAAC';
    'CFC_PAC';
    'CFC_morphPAC';
    'CFC_GC';
    'CFC_morphGC';
    'timeFeature';
    'morphTimeFeature'
    };

addpath(genpath(path_toolbox{1,1}));
for r=1:n_group
    
     path_EEG=[path_data 'group\group_' num2code(r,n_code) '\'];
    if ~exist(path_EEG,'dir')
        mkdir(path_EEG);
    end
    
    D=dir([path_EEG '*.mat']);
    if isempty(D)
        disp(['please add EEG data in ' path_data 'group\group_' num2code(r,n_code) '\']);
        keyboard;
    end
end
for r=1:size(meth_all,1)
    
    disp(['feature extraction method ' num2code(r,n_code) ' ===> ' meth_all{r,1}]);
end
rmpath(genpath(path_toolbox{1,1}));
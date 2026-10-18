tic
close all;clear all;clc;
%%

path_data='D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\dataSet_translateE1\';
path_toolbox='D:\test_matlab\toolbox\eeglab\';
%%

addpath(path_toolbox);
%%

if ~exist(path_data,'dir')
    mkdir(path_data);
end
eeglab;
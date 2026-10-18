tic
close all;clear all;clc;
%%

typeT=1; % 1 2

if typeT==1
    % unit, msec
    T=[-100,700];
elseif typeT==2
    % unit, msec
    T=[300,400];
end

n_code=3;
nGroup_old=9;

path_suffix=['(' num2str(T(1,1)) ',' num2str(T(1,2)) ')'];
path_old=['D:\test_matlab\project_EEG\data_2024_DoC\data_001_raw\temp_group' path_suffix '\'];
path_new=['D:\test_matlab\project_EEG\data_2024_DoC\data_001_raw\group' path_suffix '\'];

% consciousness level of disorders of consciousness (DoC)
% healthy control (HC)
% minmally conscious state (MCS)
% vegetative state (VS)
cl=cell(3,2);
cl{1,1}='HC';
cl{2,1}='MCS-';
cl{3,1}='VS';
cl{1,2}={
    01,'20230111_∂≠ŒƒœÈ','dwx'
    };
cl{2,2}={
    02,'20231026_ª∆≈‡«Ô1','hpq1';
    03,'20231030_¡ı–„∑Ô','lxf';
    04,'20231031_Œ‚‘£∑¢','wyf';
    09,'20231108_¡∫±Ã∑Â','lbf';
    10,'20231109_ª∆≈‡«Ô2','hpq2'
    };
cl{3,2}={
    05,'20231101_∑∂ÕÚ¡·','fwl';
    06,'20231102_¡∫”ÒÊ√','lyt';
    07,'20231104_¡ı—‡’‰1','lyz1';
    08,'20231105_◊ﬁ÷«æ¸','zzj';
    11,'20231128_¡ı—‡’‰2','lyz2';
    12,'20231129_÷‹Ω¡˙','zjl';
    13,'20231201_Î¯Ω°»À','tjr';
    14,'20231202_¿◊Õ≥∑Á','ltf'
    };

path_toolbox='D:\test_matlab\toolbox\eegGA\';
%%

addpath(genpath(path_toolbox));
%%

if ~exist(path_new,'dir')
    mkdir(path_new);
end

nState=size(cl,1);
for r1=1:nState
    for r2=1:nGroup_old
        
        rGroup_new=(r1-1)*nGroup_old+r2;
        path_oldGroup=[path_old 'group_' num2code(r2,n_code) '\'];
        path_newGroup=[path_new 'group_' num2code(rGroup_new,n_code) '\'];
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
                temp_name=D(r3).name;
                for r4=1:size(cl{r1,2},1)
                    
                    if strcmp(temp_name(1:end-4),cl{r1,2}{r4,3})
                        copyfile(path_oldGroup_temp,path_newGroup_temp);
                        disp([path_newGroup_temp ' have been done.']);
                        
                        break;
                    end
                end
            end
        end
    end
end
%%

rmpath(genpath(path_toolbox));
toc
tic
% close all;clear all;clc;
%%

% eeglab;
%%

% 1:14
rSubject=14;

% 1:5
rTrial=5;

info_subject={
    01,'20230111_∂≠ŒƒœÈ','dwx';
    02,'20231026_ª∆≈‡«Ô1','hpq1';
    03,'20231030_¡ı–„∑Ô','lxf';
    04,'20231031_Œ‚‘£∑¢','wyf';
    05,'20231101_∑∂ÕÚ¡·','fwl';
    06,'20231102_¡∫”ÒÊ√','lyt';
    07,'20231104_¡ı—‡’‰1','lyz1';
    08,'20231105_◊ﬁ÷«æ¸','zzj';
    09,'20231108_¡∫±Ã∑Â','lbf';
    10,'20231109_ª∆≈‡«Ô2','hpq2';
    11,'20231128_¡ı—‡’‰2','lyz2';
    12,'20231129_÷‹Ω¡˙','zjl';
    13,'20231201_Î¯Ω°»À','tjr';
    14,'20231202_¿◊Õ≥∑Á','ltf'
    };

info_trial={
    01,'±’—€10','EC';
    02,'’ˆ—€10','EO';
    03,'MMN','MMN';
    04,'P3A','P3A';
    05,'P300','P300'
    };

path_raw='D:\test_matlab\project_EEG\data_2024_DoC\data_001_raw\raw\';

path_old1=[path_raw 'EEG(explore)\'];
path_old2=[path_old1 info_subject{rSubject,2} '\'];
path_old3=[path_old2 info_trial{rTrial,2} '\'];


path_new1=[path_raw 'EEG(mat)\'];
path_new2=[path_new1 info_trial{rTrial,3} '\'];
path_new3=[path_new2 info_subject{rSubject,3} '.mat'];
%%

if ~exist(path_new1,'dir')
    mkdir(path_new1);
end
if ~exist(path_new2,'dir')
    mkdir(path_new2);
end

disp('load:');
disp(path_old3);
%%

nT=size(EEG.data,2);
nEvent=size(EEG.event,2);

voice_stim{1,1}=zeros(1,nT);
for r=1:nEvent
    
    voice_stim{1,1}(1,EEG.event(r).latency)=str2double(EEG.event(r).type);
end

tempEEG{1,1}=EEG.data;
EEG=[];
EEG=tempEEG;

if exist(path_new3,'file')
    disp([path_new3 ' exist, skip.']);
else
    save(path_new3,'EEG','voice_stim');
    disp([path_new3 ' have been done.']);
end
toc
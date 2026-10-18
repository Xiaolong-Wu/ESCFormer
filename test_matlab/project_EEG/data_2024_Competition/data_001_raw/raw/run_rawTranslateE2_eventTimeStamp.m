tic
close all;clear all;clc;
%%

path_dataOld='D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\dataSet_translateE1\';
path_dataNew='D:\test_matlab\project_EEG\data_2024_Competition\data_001_raw\dataSet_translateE2\';
path_toolbox='D:\test_matlab\toolbox\eegGA\';

subNo=[1 2 3 4 5 6 7 8 9];
runNo={'T'}; % {'E';'T'}

% class 1: 769, cue onset left
% class 2: 770, cue onset right
% class 3: 771, cue onset foot
% class 4: 772, cue onset tongue
% class 5: 768, start of a trail
eventType_MI=[769 770 771 772]; % [769 770 771 772 768]

fs=250;
filterBand={
    0,[45,55];
    1,[1,300];
    };
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
            EEG=[];
            load(path_dataOld_temp);
            
            temp_EEG=EEG.data;
            temp_event=floor([[EEG.event.latency]',[EEG.event.type]']);
            
            r0=0;
            for r3=1:size(temp_event,1)
                
                r0=r0+1;
                if r0>size(temp_event,1)
                    break;
                end
                
                if ~ismember(temp_event(r0,2),eventType_MI)
                    temp_event(r0,:)=[];
                    r0=r0-1;
                else
                    for r4=1:length(eventType_MI)
                        
                        if temp_event(r0,2)==eventType_MI(r4)
                            temp_event(r0,2)=r4;
                            break;
                        end
                    end
                end
            end
            
            EEG=[];
            nChannel=size(temp_EEG,1);
            for r3=1:nChannel
                
                s_filter=double(temp_EEG(r3,:));
                for r4=1:size(filterBand,1)
                    
                    if filterBand{r4,1}==0
                        s_filter_1=waveFilter(s_filter,fs,[0,filterBand{r4,2}(1,1)]);
                        s_filter_2=waveFilter(s_filter,fs,[filterBand{r4,2}(1,2),inf]);
                        s_filter=s_filter_1+s_filter_2;
                    elseif filterBand{r4,1}==1
                        s_filter=waveFilter(s_filter,fs,filterBand{r4,2});
                    end
                end
                
                EEG(r3,:)=s_filter;
            end
            Event=temp_event;
            
            save(path_dataNew_temp,'EEG','Event');
            disp([path_dataNew_temp ' have been done.']);
        end
    end
end
rmpath(genpath(path_toolbox));
toc
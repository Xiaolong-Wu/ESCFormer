tic
% close all;clear all;clc;
%%

% figureSize={1;2*[100,100,200,150]}; % scalp channel select
% figureSize={1;2*[100,100,150,100]}; % scalp or wave
figureSize={1;2*[100,100,300,50]}; % tree
%%

if figureSize{1,1}==1
    set(gcf,'position',figureSize{2,1});
end

toc
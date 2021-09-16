
clc;
clear;

start_point = 55;
end_point = 244;

log = readtable('../Results/lstm/Log_paths.csv','Format','%s%s', 'Delimiter',','); %structural/20-04-15_20-57/structural_10_5_pred.txt')
folder = log(start_point:end_point,1);
incident = log(start_point:end_point,2);

for i= 1:height(incident)
    path = strcat('../Results/lstm/',string(incident{i,:}),'/',string(folder{i,:}),'/',string(incident{i,:}),'_10_5_pred.txt');
    pred = load(path);
    %std(pred);
    fprintf('%.2f',std(pred(1:length(pred),1)));
    fprintf('\n');
end
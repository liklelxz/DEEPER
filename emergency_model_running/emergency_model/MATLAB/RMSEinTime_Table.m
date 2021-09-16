clc;
clear;

%%%%%%%%%% Parameteres we can change %%%%%%%%%%
%Buildings Names
buildingNames = {'Fire','Law','Structural','Utility','Fire','Law','Structural','Utility'};%'utility','structural','fire', 'law'}; %'weapon','od'}% 1x14 cells
%
nBuilds = size(buildingNames,2)
%1. Sequence lengths
prevStepsX=[15 15 15 15 15 15 15 15];%[10 10 10 10 10 10 10 10];% 
predStepsY= 5;%3;%
%2. Limit for axis x
uplimit = 200
%3. Models Names for legends. Real only used for predVSreal
modelNames = {'Linear','ARIMA','LSTM','Real'}% {'Linear','GCRF', 'Real'}%{'ARIMA','LSTM', 'Real'} %
modelNames_qualitative = {'Linear','ARIMA','LSTM','Real'}
%4. Colors to use in real vs predicted
colors={'y','g','b','m','r','l'}; % real is the last one
%5. Suffix of the file names
suffix = "_main";%"_gcrf_c_ch_cd_cdh";
%6. day to draw
day = 1;
%7. metric for table
metric = 'rmse'

nModels = size(modelNames,2)
%%%%%% Try not to change anything from here %%%%%%%
filePaths = fopen('pathRealVsPred15_5.txt','r') % pathRealVsPred
fileTable = fopen(strcat('../Graphs/',suffix,'.txt'),'a+');
formatString = {'%s ';'%.2f ';'\n'};
%[C{[1 2*ones(1,8) 3]}]

formatSpec = [formatString{[1*ones(1,nModels-1) 3]}] %'%s %s %s\n';%3*'%s'+'\n';
pathsMatrix = textscan(filePaths,formatSpec,'Delimiter','\t', 'headerLines', 1); %fgets

cnt = 0

for i= 1:nBuilds %building = buildingNames %
    cnt = cnt+1;
    modelPlots = cell(1,nModels)
    rmse_maeTime = cell(1,nModels-1) % real not taken into account
    tableComp = [];
    for j=1:nModels-1
        if modelNames{j} == "LSTM"
            nameFile = strcat(pathsMatrix{j}{i},"bestV_")%buildingNames{i})
            pred = load(strcat(nameFile,'pred.txt'));
        else
            nameFile = strcat(pathsMatrix{j}{i},buildingNames{i}(1:3),'_',num2str(prevStepsX(j)),'_',num2str(predStepsY))
            pred = load(strcat(nameFile,'_pred.txt'));
        end
        real = load(strcat(nameFile,'_real.txt'));
        uplimit = length(real);
        % check sanity, load other reals
        
        modelPlots{1,j} = pred
        modelPlots{1,j+1} = real ;
        %maeList = combined_RE(real, pred, 1,predStepsY)%MAE calculation - 1st, last and AVG
        modelNames{j}
%         if modelNames{j} == "ARIMA" || modelNames{j} == "Linear"
%             rmse_mae = load(strcat(nameFile,'.txt')); % rmse or mae
%         else
%             rmse_mae = load(strcat(nameFile,'_test_',metric,'.txt')); % rmse or mae
%         end
        rmse_mae = load(strcat(nameFile,'_test_',metric,'.txt')); % rmse or mae
        if modelNames{j} == "LSTM"
            fprintf(fileTable,"%f %f %f %f %f %f %f %f %f %f %f %f\n",rmse_mae);
            display(rmse_mae)
        end
        rmse_maeList = [rmse_mae(1),rmse_mae(predStepsY),mean(rmse_mae)]
        rmse_maeTime{1,j} = rmse_mae
        tableComp = [tableComp rmse_maeList];
    end
    
    % Real VS Predicted plot % do a for if want to plot for every day
      plotRealVSPredicted(modelPlots,colors,suffix, day,buildingNames{i},uplimit, modelNames,cnt) %real, pred
%       boxPlot(modelPlots{1,3},colors,suffix, day,buildingNames{i},uplimit, modelNames_qualitative)
      plotRMSEtime( rmse_maeTime, suffix , buildingNames{i} , predStepsY , modelNames ,metric,cnt)
     fprintf(fileTable,[formatString{[1*ones(1,3*nModels-3) 3]}],tableComp); %%  '%f %f %f %f %f %f %f %f %f\n'
end
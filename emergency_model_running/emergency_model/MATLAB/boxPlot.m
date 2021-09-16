clc;
clear;
buildingNames = {'utility'};%,'structural','fire', 'law'}
filePaths = fopen('pathPred10_5.txt','r');

prevStepsX = 10;
predStepsY = 5;

nBuilds = size(buildingNames,2)
formatString = {'%s ';'%.2f ';'\n'}
formatSpec = [formatString{[1*ones(1,nBuilds-1) 1]}] %'%s %s %s\n';%3*'%s'+'\n';
pathsMatrix = textscan(filePaths,formatSpec,'Delimiter','\t', 'headerLines', 1)

for i= 1:nBuilds
    nameFile = strcat(pathsMatrix{i},buildingNames{i},'_',num2str(prevStepsX),'_',num2str(predStepsY))
    pred = load(char(strcat(nameFile,'_pred.txt')));
    boxplot(pred);
    ylabel('Response Time (min)','FontSize', 18,'FontName','Times New Roman');
    xlabel('Predictions for each time step','FontSize', 18,'FontName','Times New Roman');
    
    yaxis = 0:250:1500; % Fire data
    yticks(yaxis);
    ylim([0 1500]);

    nameFile = strcat('../Graphs/boxPlot_',buildingNames{i}) %,auxBuild{1},'/' ,'predicted_v_real')
    saveas(gcf,nameFile,'epsc')
    clf
end
        
% function boxPlot(models, colors, suffix, day, building, xLimit, modelNames)
%     x = 1:xLimit;
%     nModels = size(models,2)
%     maxYAxis = 0
% %     for i=3%:nModels %star with 2 to exclude linear reg
%         modelNames
%         model = models(1:xLimit,day);
%         boxplot(models)
%         %yAxisLimit = max(model)
%         %if yAxisLimit > maxYAxis
%             %maxYAxis = yAxisLimit
%         %end
%         
%         yaxis = 0:250:1500; % Fire data
%         yticks(yaxis);
%         ylim([0 1500]); %Fire data
% %     end
% 
%     ylabel('Response Time (min)','FontSize', 18,'FontName','Times New Roman');
%     xlabel('Predictions for each time step','FontSize', 18,'FontName','Times New Roman');
%     
% %     legend boxoff;
% %     set(gca,'box','off');
% %     set(gcf, 'Position', [500, 300, 1000, 500]); %800 refers to width, 340 to height
% %     set(gca,'fontsize', 26,'FontName','Times New Roman')  ;
%     
%     nameFile = strcat('../Graphs/boxPlot_',building,char(suffix)) %,auxBuild{1},'/' ,'predicted_v_real')
%     saveas(gcf,nameFile,'epsc')
%     clf
% end
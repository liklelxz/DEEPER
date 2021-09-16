clc;
clear;


toPlot = ["trainUti","validationUti","testUti"];%"weapon","burglary","raip","od","arson","kidnap"]%

% for building = toPlot
%     
%     figure;
%     data = csvread(strcat('~/Desktop/emergency_model_running/emergency_model/Data/RemovingOutliers/BeforeProcessing/',building,'.csv'));
%     histogram(data(~isnan(data)),'BinLimits',[0 6000]);
%     title(strcat(building," before preprocessing"));
%     saveas(gcf,strcat('../Graphs/Histogram/',char(building),'_before'),'epsc');
%        
%     figure;
%     data = csvread(strcat('~/Desktop/emergency_model_running/emergency_model/Data/RemovingOutliers/AfterProcessing/',building,'.csv'));
%     histogram(data);
%     title(strcat(building," after preprocessing"));
%     saveas(gcf,strcat('../Graphs/Histogram/',char(building),'_after'),'epsc');  
% end


clc;
clear;
subtypes = ["Utility-Water Main", "Utility-Power Outage", "Utility-Other", "Utility-Water Service Line", "Utility-Steam Main", "Utility-Manhole", "Utility-Gas High Pressure", "Utility-Gas Main Rupture", "Utility-Gas Service Line", "Utility-Sewer Service"]
for i=1:10
    
    data = csvread(strcat('~/Desktop/emergency_model_running/emergency_model/Data/RemovingOutliers/SubType_before/',num2str(i),'.csv'),0,1)%Allin1Uti.csv'))%
    %data = nonzeros(data');
    figure('visible', 'off');
    boxplot(data);
    title(strcat(subtypes(i)," before preprocessing"));
    ylim([0 6000])
    saveas(gcf,strcat('../Graphs/Boxplot/',num2str(i),'before'),'epsc');
end

clc;
clear;
subtypes = ["Utility-Water Main", "Utility-Power Outage", "Utility-Other", "Utility-Water Service Line", "Utility-Steam Main", "Utility-Manhole", "Utility-Gas High Pressure", "Utility-Gas Main Rupture", "Utility-Gas Service Line", "Utility-Sewer Service"]
for i=1:10
    
    data = csvread(strcat('~/Desktop/emergency_model_running/emergency_model/Data/RemovingOutliers/SubType_after/',num2str(i),'.csv'),0,1)%Allin1Uti.csv'))%
    %data = nonzeros(data');
    figure('visible', 'off');
    boxplot(data);
    title(strcat(subtypes(i)," after preprocessing"));
    ylim([0 6000])
    saveas(gcf,strcat('../Graphs/Boxplot/',num2str(i),'after'),'epsc');
end

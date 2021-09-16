figure;
clc;
clear;
xlab = categorical({'Suspicious Package','Device', 'Other','Haz Mat'})
xlab = reordercats(xlab,{'Suspicious Package','Device', 'Other','Haz Mat'});
y = [574,104,90,23]
bar(xlab,y)
%legend({'Static','Dynamic'},'FontSize', 16,'FontName','Times New Roman','Location','northwest');hold off;legend boxoff;
set(gca, 'FontSize', 17,'FontName','Times New Roman');
xlabel('Law Sub Types', 'FontSize', 24)
ylabel('# events', 'FontSize', 24)
%     figure('visible', 'off');

saveas(gcf,strcat('../Graphs/Data/SubType/law'),'epsc');

% figure;
% clc;
% clear;
% xlab = categorical({'Electric Feeder Cable', 'Gas High Pressure','Gas Main Rupture', 'Gas Service Line', 'Manhole', 'Power Outage', 'Water Main', 'Water Service Line' 'Other'})
% xlab = reordercats(xlab,{'Electric Feeder Cable', 'Gas High Pressure','Gas Main Rupture', 'Gas Service Line', 'Manhole', 'Power Outage', 'Water Main', 'Water Service Line' 'Other'});
% y = [16,80,54,50,78,257,401,114,149]
% bar(xlab,y)
% %legend({'Static','Dynamic'},'FontSize', 16,'FontName','Times New Roman','Location','northwest');hold off;legend boxoff;
% set(gca, 'FontSize', 17,'FontName','Times New Roman');
% xlabel('Utility Sub Types', 'FontSize', 24)
% ylabel('# events', 'FontSize', 24)
% %     figure('visible', 'off');
% 
% saveas(gcf,strcat('../Graphs/Data/SubType/utility1'),'epsc');


% figure;
% clc;
% clear;
% xlab = categorical({'Electric', 'Gas', 'Manhole', 'Power', 'Water' 'Other'})
% xlab = reordercats(xlab,{'Electric', 'Gas', 'Manhole', 'Power', 'Water' 'Other'});
% y = [16,184,78,257,515,149]
% bar(xlab,y)
% %legend({'Static','Dynamic'},'FontSize', 16,'FontName','Times New Roman','Location','northwest');hold off;legend boxoff;
% set(gca, 'FontSize', 17,'FontName','Times New Roman');
% xlabel('Utility Sub Types', 'FontSize', 24)
% ylabel('# events', 'FontSize', 24)
% %     figure('visible', 'off');
% 
% saveas(gcf,strcat('../Graphs/Data/SubType/utility2'),'epsc');

figure;
clc;
clear;
xlab = categorical({'Water Main', 'Power Outage', 'Other', 'Water Service Line', 'Gas', 'Manhole'})
xlab = reordercats(xlab,{'Water Main', 'Power Outage', 'Other', 'Water Service Line', 'Gas', 'Manhole'});
y = [401,257,149,114,80,78]
bar(xlab,y)
%legend({'Static','Dynamic'},'FontSize', 16,'FontName','Times New Roman','Location','northwest');hold off;legend boxoff;
set(gca, 'FontSize', 17,'FontName','Times New Roman');
xlabel('Utility Sub Types', 'FontSize', 24)
ylabel('# events', 'FontSize', 24)
%     figure('visible', 'off');

saveas(gcf,strcat('../Graphs/Data/SubType/utility'),'epsc');

figure;
clc;
clear;
xlab = categorical({'2nd Alarm', '1st Alarm', '3rd Alarm', 'Commercial', 'Other' 'Haz Mat'})
xlab = reordercats(xlab,{'2nd Alarm', '1st Alarm', '3rd Alarm', 'Commercial', 'Other' 'Haz Mat'});
y = [642,236,183,93,89,81]
bar(xlab,y)
%legend({'Static','Dynamic'},'FontSize', 16,'FontName','Times New Roman','Location','northwest');hold off;legend boxoff;
set(gca, 'FontSize', 17,'FontName','Times New Roman');
xlabel('Utility Sub Types', 'FontSize', 24)
ylabel('# events', 'FontSize', 24)
%     figure('visible', 'off');

saveas(gcf,strcat('../Graphs/Data/SubType/fire'),'epsc');

figure;
clc;
clear;
xlab = categorical({'Scaffold', 'Collapse', 'Other', 'Construction', 'Stability' 'Partial Collapse'})
xlab = reordercats(xlab,{'Scaffold', 'Collapse', 'Other', 'Construction', 'Stability' 'Partial Collapse'});
y = [125,105,102,101,85,69]
bar(xlab,y)
%legend({'Static','Dynamic'},'FontSize', 16,'FontName','Times New Roman','Location','northwest');hold off;legend boxoff;
set(gca, 'FontSize', 17,'FontName','Times New Roman');
xlabel('Utility Sub Types', 'FontSize', 24)
ylabel('# events', 'FontSize', 24)
%     figure('visible', 'off');

saveas(gcf,strcat('../Graphs/Data/SubType/structural'),'epsc');


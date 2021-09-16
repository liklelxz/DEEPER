clc;
clear;

% Building names
toPlot = ["fire", "law", "structural", "utility"];%"weapon","burglary","raip","od","arson","kidnap"]%

% endpoint = 2953;
start_point = 1;
endpoint = 600;

for building = toPlot
    
    data = csvread(strcat('~/Desktop/emergency_model_running/emergency_model/Data/With_Features/',building,'.csv'),1,2);%With_Features %911
    endpoint = length(data);
    y=data(start_point:endpoint,1);
    length(y);
    
    % Time steps
    x = start_point:endpoint;%1:48;%
    
    % To display graph in MATLAB, set value of visible as 'on'
    figure('visible', 'off');

    line_width = 1;
    marker_size = 0.1;

    plot(x,sgolayfilt(y,3,11),'d-','MarkerFaceColor',[0.2706 0.3412 0.8784],'LineWidth',line_width,'MarkerSize',marker_size,'Color',[0.2706 0.3412 0.8784],'LineStyle','-');hold on;

    set(gca, 'FontSize', 24,'FontName','Times New Roman');
    
%     % Time steps
%     set(gca,'XLim',[1 output_sequence],'XTick',[1:output_sequence]);
    set(gca,'XLim',[start_point endpoint]);%[1 48]);%,'XTick',[1:endpoint]);
% 
    xlabel('Timesteps', 'FontSize', 26);%,'FontName','Times New Roman');
   h = ylabel('Response Time', 'FontSize', 26);%,'FontName','Times New Roman');
%    rect = [0.15 105.5000 15];

%     set(h, 'Position', rect)
%     set(gca,'xtick',[1,2,3,4,5,6])
%     set(gca,'xticklabel',{20,40,60,80,100,120})
    
%     title(strcat('Building ',building));


    yaxis = 0:500:3000;
    yticks(yaxis);
    ylim([0 3000]);
    
    fprintf(building);
    fprintf('  Median: %.2f ',median(y));
    fprintf('Mean: %.2f ',mean(y));
    fprintf('Max: %.2f ',max(y));
    fprintf('Min: %.2f ',min(y));
    fprintf('Zeroes: %.2f ',nnz(~y));
    
    fprintf('\n');
% 
%     % Save figure in the current folder as building_name.eps    
     saveas(gcf,strcat('../Graphs/Data/',char(building)),'epsc');
     
     x0=10;
y0=10;
width=1000;
height=500;
set(gcf,'units','points','position',[x0,y0,width,height]);

end






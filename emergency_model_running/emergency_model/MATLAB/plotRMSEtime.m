function plotRMSEtime(models, suffix, building, ySteps, modelNames,metric,cnt)
    colors = [0.2196 0.2235 0.2275; 0.2706 0.3412 0.8784; 0.8980 0.2039 0.2039 ;0.2039 0.5686 0.2471; 0.2706 0.3412 0.8784]
    shapes = {'d-', 's-.','o--','*--'}
    nModels = size(models,2)
    x = 1:ySteps;
    for i=1:nModels %models
        model = models{1,i};
        plot(x, model.', shapes{1,i},'MarkerFaceColor', colors(i,1:3),'LineWidth',2,'MarkerSize',8,'Color',colors(i,1:3));hold on;
    end
    
    legend(modelNames, 'FontSize', 18,'FontName','Times New Roman','Location','northwest');hold off;
    legend boxoff;
    set(gcf, 'Position', [500, 300, 1000, 500]);
    set(gca, 'FontSize', 24,'FontName','Times New Roman');
    
    % Time steps
    set(gca,'XLim',[1 ySteps],'XTick',[1:ySteps]);

    xlabel('Timesteps', 'FontSize', 28);%,'FontName','Times New Roman');
    if metric == "rmse"
        ylabel('RMSE', 'FontSize', 28);%,'FontName','Times New Roman');
    else
        ylabel('MAE', 'FontSize', 28);%,'FontName','Times New Roman');
    end
    title(strcat(building));

    x0=10;
y0=10;
width=510;
height=390;
set(gcf,'units','points','position',[x0,y0,width,height]);
    
    % Save figure in the current folder as building_name.eps    
    saveas(gcf,strcat('../Graphs/',char(building),char(suffix),'_',metric,'_',num2str(cnt)),'epsc');%strcat(building,'_rmse_lstm_chw')
end
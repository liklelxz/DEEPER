function plotRealVSPredicted(models, colors, suffix, day, building, xLimit, modelNames,cnt)
    if building == "Fire"
        xLimit = 200;
    end
    x = 1:xLimit;
    nModels = size(models,2)
    maxYAxis = 0
    for i=1:nModels %star with 2 to exclude linear reg
        model = models{1,i}(1:xLimit,day);
        plot(x,sgolayfilt(model.',3,11) , colors{1,i},'LineWidth',2,'MarkerSize',2);hold on;%sgolayfilt(model.',3,11)
        yAxisLimit = max(model)
        if yAxisLimit > maxYAxis
            maxYAxis = yAxisLimit
        end
    end

    ylabel('Response Time (min)','FontSize', 18,'FontName','Times New Roman');
    xlabel(strcat('Prediction for each test sample'),'FontSize', 18,'FontName','Times New Roman');
    xticks(0:50:xLimit);%0:xLimit:10
    yaxis = 0:400:3200; % Fire data
    yticks(yaxis);
    ylim([10 3200]); %Fire data
    title(strcat(building));

%     yaxis = 0:40:200; % 911 data
%     yticks(yaxis);
%     ylim([10 200]); %911 data
    
    legend(modelNames,'FontSize', 20,'FontName','Times New Roman');

    legend boxoff;
    set(gca,'box','off');
    set(gcf, 'Position', [500, 300, 1000, 500]); %800 refers to width, 340 to height
    set(gca,'fontsize', 26,'FontName','Times New Roman')  ;
    
    nameFile = strcat('../Graphs/',num2str(day),building,char(suffix),'_',num2str(cnt)) %,auxBuild{1},'/' ,'predicted_v_real')
    saveas(gcf,nameFile,'epsc')
    clf
end
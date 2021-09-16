data = readtable("Sn.csv");
x = 0:length(data.ResponseTime)-1;
y = data.ResponseTime;

h = plot(x,y);
ax = ancestor(h, 'axes');
ax.YAxis.Exponent = 0;
ytickformat('%.0f');
box off



formatout = 'yyyy-mm';
i = 1;
ans = xticks;
str = {}

while(i<length(ans))
    str(end +1)= cellstr(datestr(data.CreationDate(ans(i)+1),formatout));
    i = i + 1;
end



xticklabels(str)

ylim([0,120000]);
set(gca,'TickDir','out')
set(h,'LineWidth',0.1)
xlabel("Creation Date")
ylabel("Response Time")
pbaspect([8 1 1])
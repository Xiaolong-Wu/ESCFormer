function barChart(groupData,groupName,barChartMarkerSizeData,barFaceColor,barEdgeColor,markerFaceColor,markerEdgeColor,barChartLabel_y)

barWidth=0.6;
lineWidth=2;

lineWidthError=1;
markerSizeError=5;

lineWidthData=2;
%%

nGroup=size(groupData,2);

x=zeros(1,nGroup);
e=zeros(1,nGroup);
for r=1:nGroup
    
    x(1,r)=mean(groupData{1,r});
    e(1,r)=std(groupData{1,r},0,1);
end

bar(x,barWidth,'FaceColor',barFaceColor,'EdgeColor',barEdgeColor,'LineWidth',lineWidth);
for r=1:nGroup
    
    hold on;
    plot([r r],[x(1,r)-e(1,r) x(1,r)+e(1,r)],'k','LineWidth',lineWidthError);
    hold on;
    plot(r,x(1,r)-e(1,r),'o','MarkerSize',markerSizeError,'MarkerFaceColor','k','MarkerEdgeColor','k');
    hold on;
    plot(r,x(1,r)+e(1,r),'o','MarkerSize',markerSizeError,'MarkerFaceColor','k','MarkerEdgeColor','k');
end
set(gca,'XTick',[1:nGroup],'XTickLabel',groupName);

if barChartMarkerSizeData>0
    for r1=1:nGroup
        
        nSubject=size(groupData{1,r1},1);
        for r2=1:nSubject
            
            x=r1+(-1)^r2*rand(1,1)*barWidth/2;
            y=groupData{1,r1}(r2,1);
            plot(x,y,'o','MarkerSize',barChartMarkerSizeData,'MarkerFaceColor',markerFaceColor,'MarkerEdgeColor',markerEdgeColor,'LineWidth',lineWidthData);
        end
    end
end

xlim([0.5,nGroup+0.5]);
if sum(barChartLabel_y)~=0
    ylim(barChartLabel_y);
end
end
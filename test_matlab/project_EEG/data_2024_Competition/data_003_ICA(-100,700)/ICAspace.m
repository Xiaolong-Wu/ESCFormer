function d=ICAspace(xb)
% xb=[x1,x2,x3,bregma]
% d=sum(||x(1:3)-xi|-w|)

global coordinate
global temp_Wchannel_s

nChannel=size(coordinate,1);
d_all=zeros(nChannel,1);

for r=1:nChannel
    
    d_all(r,1)=((((xb(1:3)-[coordinate(r,:),0])*(xb(1:3)-[coordinate(r,:),0])')^0.5-xb(4)*temp_Wchannel_s(1,r))^2)^0.5;
end

d=mean(d_all);
end
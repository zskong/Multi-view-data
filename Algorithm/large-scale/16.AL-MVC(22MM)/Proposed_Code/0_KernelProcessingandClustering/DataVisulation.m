function [] = DataVisulation(Data , Y , varargin)

SN = length(Y);
CluNum = length(unique(Y));
Label = unique(Y);
color = ['r' , 'g' , 'b' , 'c' , 'm' , 'y' , 'k'];

for I = 1 : CluNum
    CI =  Y == Label(I) ;
    plot3(Data(CI , 1) , Data(CI , 2) ,Data(CI , 3), strcat(color(I) , 'o'));
    grid on
    hold on
end

if ~isempty(varargin)
    ExtraI = varargin{1};
    plot(Data(ExtraI , 1) , Data(ExtraI , 2) , strcat(color(CluNum+1) , 'o'));
end
end
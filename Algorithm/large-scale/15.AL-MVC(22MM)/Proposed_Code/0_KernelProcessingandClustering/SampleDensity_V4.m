function [D , MinD] = SampleDensity_V4(KH , CluNum)
[SNum , ~ , KerNum] = size(KH);
D = zeros(KerNum , SNum);
SSN = round(SNum / CluNum * 2);

for ii = 1 : KerNum
    TKH = KH(:,:,ii);
    TKH = sort(TKH , 'descend');
    TKH = TKH(1 : SSN , :);
    TKHMax = repmat( max(TKH) , SSN , 1 );
    TKHMin = repmat( min(TKH) , SSN , 1 );
    TKH = ( TKH - TKHMin ) ./ ( TKHMax - TKHMin );
    TKH = TKH ./ repmat(sum(TKH) , SSN , 1);
    Index = TKH == 0;
    TKH(Index) = 10^(-6);
    D(ii , :) = -sum(TKH .* log(TKH)) / log(SSN);
end
MinD = min(D);
end
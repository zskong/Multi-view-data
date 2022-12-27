function MinD = CExtent(MinD)

MD = 0.85;

index = MinD < MD;
MinD(index) = 1;
index = logical(1 - index);
TotalIndex = 1 : length(MinD);
index = TotalIndex(index);
MI = min(MinD(index));
MXI = max(MinD(index));
MinD(index) = ( MinD(index) - MI ) / (MXI - MI) * 5;
MinD(index) = (2 * ones(1 , length(index) ) ./ (1 + exp( -MinD( index ))) - 1) *0.5;
MinD(index) = MinD(index) + 1;
end
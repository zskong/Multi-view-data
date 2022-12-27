function D = SampleDensity_V2(KH , k)
[SNum , ~ , KerNum] = size(KH);
D = zeros(KerNum , SNum);

for ii = 1 : KerNum
    TKH = KH(:,:,ii);
    MS = sum(sum(TKH)) / SNum^2;
    if MS >=0
        II = TKH > MS * k;
    else
        II = TH > MS / k;
    end
    D(ii,:) = ones(1 , SNum) * II;
end

DMin = min(D , [] , 2);
DMax = max(D , [] , 2);
DMin = repmat(DMin , 1 , SNum);
DMax = repmat(DMax , 1 , SNum);
Diff = DMax - DMin;
D = (D - DMin) ./ Diff;

end
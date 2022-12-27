function NewM = NewMCalculation(Km , L , H)
KerNum = size(Km , 3);
NewM = zeros(KerNum , KerNum);

for ii = 1 : KerNum
    for jj = 1 : KerNum
        NewM(ii,jj) = trace(H' * Km(:,:,ii) * L * Km(:,:,jj)*H);
    end
end

end
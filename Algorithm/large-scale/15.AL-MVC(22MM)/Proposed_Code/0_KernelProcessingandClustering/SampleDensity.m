function D = SampleDensity(KH , k)
[SNum , ~ , KerNum] = size(KH);
D = zeros(KerNum , SNum);

for ii = 1 : KerNum
    TKH = KH(:,:,ii);
    % 用2范数归一化
    aa = sqrt(diag(TKH * TKH'));
    TKH = TKH ./ repmat(aa' , SNum , 1);
    %         % 用一范数归一化
    %         aa = sum(abs(TKH));
    %         TKH = TKH ./ repmat(aa , SNum , 1);
    
    TKH = sort(TKH , 'descend');
    D(ii , :) = sum( TKH(2 : k+1 , : ) );
end

end
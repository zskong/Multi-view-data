function ind = node_sampling_dblp(A, m, alpha)

D0 = sum(A{1}, 2);
D1 = sum(A{2}, 2);
D2 = sum(A{3}, 2);
D =  D0(:) + D1(:) + D2(:);

D = D.^alpha;
D = D./10000;

tot = sum(D);
% disp(tot);
p = D./tot;
% disp(size(p));
for i=1:size(p,1)-1
    p(i+1) = p(i+1) + p(i);
end

vis = zeros(size(D,1),1);
% disp(size(vis));
ind = [];
while m > 0
    while 1
        rd = rand;
        pos = lower_bound(p, rd);
        if vis(pos) == 1
            continue
        else
            vis(pos) = 1;
            ind = [ind pos];
            m = m - 1;
            break
        end
    end
end

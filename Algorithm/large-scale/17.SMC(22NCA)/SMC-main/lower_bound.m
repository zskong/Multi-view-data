function l = lower_bound(p,rd)
l = 0;
r = size(p,1) - 1;
while l < r
    mid = fix(( l + r )/2);
    if p(mid) > rd
        r = mid;
    else
        l = mid + 1;
    end
end
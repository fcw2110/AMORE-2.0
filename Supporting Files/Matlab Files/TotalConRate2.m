function total_prod_rate = TotalConRate2(Spname,S)
f = ExtractRates(Spname,S,1);
if length(f)>0

    fl = length(f(:,1));
    sum_list = [];
    for i = 1:fl-1
        s = sum(min(0,f(i,:)))*(S.Time(i+1)-S.Time(i));
        sum_list = [sum_list s];
    
    end
    total_prod_rate = sum(sum_list);
else
    total_prod_rate= 0;
end
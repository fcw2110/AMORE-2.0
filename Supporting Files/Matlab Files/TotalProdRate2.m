function total_prod_rate = TotalProdRate2(Spname,S)


f = ExtractRates(Spname,S,0);

if length(f)>0

    fl = length(f(:,1));
    sum_list = [];
    for i = 1:fl-1
        %f(i,:)
        s = sum(max(0,f(i,:)))*(S.Time(i+1)-S.Time(i));

        %max(0,f(i,:))
        sum_list = [sum_list s];


    end
    total_prod_rate = sum(sum_list);
else
    total_prod_rate= 0;
end





% 
% function total_prod_rate = TotalProdRate2(Spname,S)
% 
% 
% f = ExtractRates(Spname,S,1);
% 
% if length(f)>0
% 
%     fl = length(f(:,1));
%     fl2 = length(f(1,:));
%     sum_list = [];
% 
%     for i = 1:fl2
%         sumy = 0;
%         for j = 1:fl-1
%         %f(i,:)
%             sumy = sumy + max(0,f(j,i))*(S.Time(j+1)-S.Time(j));
% 
%         %max(0,f(i,:))
% 
%         end
%         sum_list = [sum_list sumy];
% 
% 
%     end
%     % = sum_list;
% 
%     total_prod_rate = sum(sum_list);
% else
%     total_prod_rate= 0;
% end
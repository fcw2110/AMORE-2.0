function [Fitness]=Fitness_species_input_6_3_24(reference,test, species1, species2, weights, consumption)
% inputs: reference mechanism data, test mechanism data, reference mechanism species to be compared
% test mechanism species to be compared
% compared species weights
% there are different lists of species for each mechanism in the case that the species being compared have 
% different names, or if a group of species is being measured
% weights: the weighting for each species. List with same length as species
% consumption: If consumption needs to be taken into consideration for
% that species. List with same length as species
L = length(species1);
error = 0;
results = [];

for i=1:L
	if size(species1{i},1)==1
        species_ref_con = TotalConRate2(species1{i}, reference);
		species_ref_prod = TotalProdRate2(species1{i}, reference);
    else
        L2 = length(species1{i});
        species_ref_prod = 0;
        species_ref_con = 0;
        for j=1:L2
            Cnames = reference.Cnames;
            [tf,iSp] = ismember(species1{i}(j),Cnames);
            if ~tf
                species_ref_prod = species_ref_prod;
                species_ref_con = species_ref_con;
            else
                species_ref_prod = species_ref_prod + TotalProdRate2(species1{i}(j), reference);
                species_ref_con = species_ref_con + TotalConRate2(species1{i}(j), reference);
            end
        end

    end
    if size(species2{i},1)==1
		species_test_prod = TotalProdRate2(species2{i}, test);
        species_test_con = TotalConRate2(species2{i}, test);
    else
        L2 = length(species2{i});
        species_test_prod = 0;
        species_test_con = 0;
        for j=1:L2
            
            Cnames = test.Cnames;
            [tf,iSp] = ismember(species1{i}(j),Cnames);
            if ~tf
                species_test_prod = species_test_prod;
                species_test_con = species_test_con; 
            else
                species_test_prod = species_test_prod + TotalProdRate2(species2{i}(j), test);
                species_test_con = species_test_con + TotalProdRate2(species2{i}(j), test);
            end
        end
    end
    if consumption(i)==1
        test_net = species_test_prod + species_test_con;
        ref_net = species_ref_prod + species_ref_con;
    else
        test_net = species_test_prod;
        ref_net = species_ref_prod;
    end
    
    error = error + weights(i)*abs(ref_net-test_net)/max((abs(ref_net) + abs(test_net)),1e-20);
    %results = [results; error];%[results; weights(i)*abs(ref_net-test_net)/max((abs(ref_net) + abs(test_net)),1e-20)];
    %results = [results; abs(ref_net-test_net)/max((abs(ref_net) + abs(test_net)),1e-20)];
    results = [results; test_net];

end

%Fitness = error/sum(weights);
%results
Fitness = results;


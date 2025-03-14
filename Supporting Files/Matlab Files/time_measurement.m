clear
S = [];
samples = csvread('samples8.csv');

runs = ["caltech_ref_fixes_6_1", "test_300_destiff","test_275_destiff","test_250_destiff","test_225_destiff","test_200_destiff","test_180_destiff","test_160_destiff","test_150_destiff","test_140_destiff","test_135_destiff","test_130_destiff","test_120_destiff","test_110_destiff","test_100_destiff","test_90_destiff","test_80_destiff","test_70_destiff","test_60_destiff","test_55_destiff","test_50_destiff","test_48_destiff","test_46_destiff" ,"test_44_destiff","test_43_destiff","test_42_destiff"]%"time_test_mech","mini_speed_test", "test_100_speed"]%,"f0am_AMORE_newest_stuffn_35_ao","f0am_AMORE_newest_stuffn_35_go","f0am_AMORE_newest_stuffn_35_g","f0am_optimized_mechanism_35a_opt","f0am_optimized_mechanism_35ao_opt","f0am_optimized_mechanism_35go_opt","f0am_optimized_mechanism_35g_opt","f0am_optimized_mechanism40_ugs","f0am_optimized_mechanism40_n","f0am_optimized_mechanism40_sgd","f0am_optimized_mechanism40_random_ugs","f0am_AMORE_newest_stuffn_40"]
runs = ["test_55_destiff","test_50_destiff","test_48_destiff","test_46_destiff" ,"test_44_destiff","test_43_destiff","test_42_destiff"]%"time_test_mech","mini_speed_test", "test_100_speed"]%,"f0am_AMORE_newest_stuffn_35_ao","f0am_AMORE_newest_stuffn_35_go","f0am_AMORE_newest_stuffn_35_g","f0am_optimized_mechanism_35a_opt","f0am_optimized_mechanism_35ao_opt","f0am_optimized_mechanism_35go_opt","f0am_optimized_mechanism_35g_opt","f0am_optimized_mechanism40_ugs","f0am_optimized_mechanism40_n","f0am_optimized_mechanism40_sgd","f0am_optimized_mechanism40_random_ugs","f0am_AMORE_newest_stuffn_40"]

runs = ["caltech_ref_fixes_6_1","test_70_stiff","test_60_stiff","test_55_stiff","test_50_stiff","test_48_stiff","test_46_stiff" ,"test_44_stiff","test_43_stiff","test_42_stiff"]%"time_test_mech","mini_speed_test", "test_100_speed"]%,"f0am_AMORE_newest_stuffn_35_ao","f0am_AMORE_newest_stuffn_35_go","f0am_AMORE_newest_stuffn_35_g","f0am_optimized_mechanism_35a_opt","f0am_optimized_mechanism_35ao_opt","f0am_optimized_mechanism_35go_opt","f0am_optimized_mechanism_35g_opt","f0am_optimized_mechanism40_ugs","f0am_optimized_mechanism40_n","f0am_optimized_mechanism40_sgd","f0am_optimized_mechanism40_random_ugs","f0am_AMORE_newest_stuffn_40"]

% "test_300_stiff","test_275_stiff","test_250_stiff","test_225_stiff","test_200_stiff","test_180_stiff","test_160_stiff","test_140_stiff","test_120_stiff","test_100_stiff","test_90_stiff","test_80_stiff",

Lr = length(runs)


for k = 1:Lr
    for p = 1:1
       
        ChemFiles = {...
            'F0AM_isop_K_update(Met)';
            'F0AM_isop_J(Met,1)';
            runs(k);
            };
        for jj = 1:6
                Met = {...
        %  		names       values          
    		    'P'         1000          ; %Pressure, mbar
    		    'T'         292                ; %Temperature, K
    		    'RH'        0                ; %Relative Humidity, percent
    		    'LFlux'     'ExampleLightFlux.txt'     ; %Text file for radiation spectrum
                'SUN'       samples(jj,8)             ;
    		    'jcorr'     samples(jj,8)              ; %light attenuation factor
    		    };
            InitConc = {...
    %   		names       conc(ppb)           HoldMe
        	    'O2'        210000000             1;
        	    'ISOP'      1       0;%samples(jj,1)         0;
        	    'NO3'       samples(jj,7)         1;
    		    'OH'        samples(jj,2)         1; %10^6 molecules per CCH0         0;
                'O3'        samples(jj,6)         1;
                'NO'        samples(jj,5)         1;
        	    'HO2'       samples(jj,3)         1;
                'CH3CO3'    samples(jj,9)         1;
                'CH3OO'     samples(jj,9)         1;
                %'NO2'       0                     1;
        	    };
            BkgdConc = {...
    %   names           values
            'DEFAULT'       0;   %0 for all zeros, 1 to use InitConc
            };
    
          
            ModelOptions.Verbose       = 0;
            ModelOptions.EndPointsOnly = 0;
            ModelOptions.LinkSteps     = 0;
            ModelOptions.IntTime       = 24*3600;
            ModelOptions.SavePath      = 'ChamberExampleOutput.mat';
            ModelOptions.GoParallel    = 0;
            tic
            for n = 1:30
                x = F0AM_ModelCore(Met,InitConc,ChemFiles,BkgdConc,ModelOptions);
            end
            toc
        end
    end
   
    
end




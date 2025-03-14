The algorithm and necessary functions are contained within the AMORE_v2.py file. Sample jupyter notebooks for isoprene (AMORE_ISOP_TESTING_1_29_25.ipynb) and camphene (AMORE_CAMPHENE_workbook_clean.ipynb) are provided. The algorithm is imported into these workbooks using:
import AMORE_v2 as am

The main reduction function is called using am.AMORE_mechanism_reduction(mechanism, background_species, conditions, settings)

The algorithm requires that the full mechanism be in the correct format. This format is a custom python based format which follows the following definition:
class Mechanism:
    def __init__(self, species, reactions):
        self.species = species
        self.reactions = reactions

class Reaction:
    def __init__(self, reactants, prod_dict, rate_law, eval_rate_law, rate, rate_string = '', multiplier = 1):
        self.reactants = reactants
        self.prod_dict = prod_dict
        self.rate_law = rate_law
        self.eval_rate_law = eval_rate_law
        self.rate = rate
        self.rate_string = rate_string
        self.multiplier = multiplier

A mechanism is an object class composed of a list of species and list of reactions. 
A reaction is an object class composed of:
- .reactants: a list of the reactants
- .prod_dict: a dictionary of the products with keys as the species and values as the stoichiometric coefficients
- .rate_law: a string of the rate law function or otherwise 'null' if using eval_rate_law
- .eval_rate_law: a list of the evaluated rate laws for each condition, eg [eval1, eval2, eval3...]
- .rate: set this to 1.
- .rate_string: this is used for GECKO-A postprocessing to keep the format of the GECKO-A mechanism rate law by inputting it as a string
- .multiplier: this is used in the algorithm but starts at one as the default, can be left blank

Note: The AMORE 2.0 algorithm requires all rate constants to be evaluated at each condition. This can be done within the algorithm assuming that all 'rate_law' strings for each reaction in the mechanism can be evaluated, or they can be evaluated outside of the algorithm and input directly as 'eval_rate_law', the number of evaluated rate laws for each reaction must match the number of input conditions. If using 'eval_rate_law', input 'null' for the rate_law. Otherwise the AMORE algorithm will try to evaluate the rate_law. If not using eval_rate_laws, can put anything as a placeholder as they will be ignored as long as 'rate_law' is not 'null'. 


The user should create reactions using the following command:

am.Reaction(reactants, prod_dict, rate_law, eval_rate_law, 1, rate_string)

With a list of reactions, the user can create a mechanism using the following command:

am.Mechanism(species_list, Reactions)



Regardless of the oiginal format of the full mechanism, it must be converted to this format. 
The notebooks for isoprene and camphene demonstrate this for two different initial mechanism file types.


Inputs to the algorithm (in this order):
- the mechanism in the above format
- background_species: a list of species that will default as secondary reactants when calculating relative rates, we usually use HOx, NOx, Ox species, and any other inorganic reactants or small organic radicals
- conditions: a list of dictionaries (one dictionary for each condition) containing meteorological parameters and species concentrations needed to calculate the relative rates of each reaction
- settings: a mix of optional and required settings for the reduction to proceed, listed below

Condition Formatting: 
conditions = [condition1, condition2.....]
condition1 = {'temp':298, 'pressure': 1000, 'OH': 1, 'HO2': 0.007,'sza': 0, 'HO': 0.0002,'NO3': 0.0,'NO': 0.02,'O2': 210000000,'CMPHEN': 1}
Note: temperature, sza and pressure must be listed as 'temp', 'sza' and 'pressure'. Species should appear as named in the mechanism and concentrations should be in units of ppb. 

Settings:
Settings is formatted as a dictionary with keys as the setting and values as the value of that settings. Some are optional and some are not optional.

Required settings:
'roots': a list of root species for the mechanism. Eg: ['ISOP'], or for a mechanism with multiple roots: ['APINENE','CMPHEN']

'Mechanism Size': the desired mechanism size in terms of number of species. This will count all species in the species list including background species and priority species. 
For example, if you have a mechanism with 400 species, and only 350 of them are eligible to be removed, setting the mechanism size to 51 will result in a mechanism with only one species remaining among those that were eligible to be removed.

Optional settings:
'Protected': a list of species that should not be removed from the mechanism, eg: ['HCHO','MGLY','PAN']
default = []

'Categories': A list of categories ([category1, category2....]) with each category having the format: ['name',[list of species in the category]].
default = []

'Manual Groups': A list of groups ([group1, group2, group3...]) with each group having the format: [list of species in the group]. Eg: IEPOX group: ['ISOP1OH23O4OHt','ISOP1OH23O4OHc', 'ISOP1OH2OH34O'].
This selection will override any groups identified by the AMORE algorithm if there is overlap in selection. Species in groups will be merged with the highest priority species in the group if they are removed. To ensure that the group remains in the mechanism, name one species in the group in the Protected list. 
default = []

'No Group': A list of species that should not be grouped. Format: [list of species not to be grouped]. We have observed that the AMORE 2.0 group selection occasionally selects species that should not be grouped together, leading to increased error. The user can correct this by identifying the species that should not have been grouped and naming it here. Alternatively, if the user wants to mute the grouping algorithm altogether, simply list every species here and no groups will be created. Or, explicitly state all groups in the manual groups setting and then name the remaining species here. 
default = []

'Remove Species': The AMORE 2.0 algorithm has a default method for determining the order in which to remove species, and it has functioned well thus far for us. Here, you can specify species to be removed, and they will be removed first. Between this and the Protected list, the user can specify exactly which species should remain in the mechanism and which should be removed.
default = []

'Background Rxns': If the mechanism contains reactions that should not be taken into account, include them here. The format is [rxn index 1, rxn index 2, ...], where the index listed indicates the position of the background reaction in the mechanism.reactions list. Generally, reactions involving inorganic species (eg: H2O2 --> OH + OH) should be included in this list. 
default = []

'Aerosol Rxns': Specifically for GECKO-A mechanisms, to identify aersol phase reactions that will be ignored, same format as Background Rxns'. The AMORE algorithm can handle heterogeneous chemistry so long as the rate constants are evaluatable. We did not incorporate aerosol phase rate constant evaluation into our GECKO-A camphene reductions, which is why these reactions are ignored. 
default = []

'Iterations': The number of iterations to do for optimization of rate constants for species in strongly connected components. This rate constant optimization uses gradient descent and improves the accuracy of strongly connected components where a portion of the species have been removed. However, it takes a long time to run. For our isoprene mechanisms, it takes up to 30 minutes at 100 iterations. For our camphene mechanisms, one iteration took up to several hours. For this reason, we do not recommend this optimization for large GECKO-A mechanisms, but if desired, it will likely take over one day. The recommended number of iterations is no less than 50 and up to 200. Performance progress will be printed at each iteration. 
default = 0

'Reduce Stiffness': True or False. If true, the rate constants of the fastest reactions will be reduced where appropriate. This is done to reduce the stiffness of the mechanism. 
default = False

'Stiffness Threshold': Number specifying order of magnitude (eg, 3,4, or 5). The order of magnitude difference in rates needed to flag a reaction as stiff, relative to the median rate of the mechanism. For example, if the median rate is 10^-5 and the relative rate of a given reaction is 10^-1, a stiffness threshold of 5 will not identify this reaction as stiff, whereas a threshold of 3 will. We recommend a threshold of 5. 
No default, must be supplied if Reduce Stiffness is True. 

'Reference Rate': If the user wants to use a different reference rate for stiff reaction identification in place of the median, this can be specified here as a float. 
default = None

'Remove Reactions': True or False. If True, reactions with identical reactants will be merged. Eg: ISOP + OH --> ISOP1OHt, ISOP + OH --> ISOP1OHc becomes ISOP + OH --> A* ISOP1OHt + B* ISOP1OHc, where A and B are determined from the relative rates of the two merged reactions. If False, reactions will not be merged, however, this will result in many more reactions in the mechanism. 
default = True

'Keep Cycle Reactions' = True or False. Reactions involving reactants and products of species that are in a strongly connected component often do not merge accurately. If True, these reactions will not be merged. We recommend keeping as True.
default = True

'Remove Weak Rxns': True or False. If True, weak reactions relative to other reactions for a given species will be removed. This feature needs improvement, so we recommend keeping it as False until further updates are made. If the user needs to remove slow reactions, this may be feasible to do manually as reduced mechanisms are comparatively small. 
default = False

'Weak Reaction Cutoff': Number indicating how much slower a reaction needs to be relative to other reactions of a given species before it is removed. Example, if ISOP has two reactions, ISOP + OH and ISOP + NO3, with relative rates ROH and RNO3, if RNO3< cutoff*ROH, and 'Remove Weak Rxns' is True, then ISOP + NO3 will be removed. We recommend a cutoff less than 0.01. 
Must be supplied if 'Remove Weak Rxns' is true. 

'No Counts': A set of species {spec1, spec2...}. For GECKO-A reactions, there are phenomenon (eg photolysis indicated by '+ HV', and special rates specified by '+ EXTRA') that are listed as reactants but are not actual reactants. The relative rate calculation depends on the number of reactions, so these markers must be specified as not counting as actual reactants. For GECKO-A mechanisms, we recommend the following settings:
settings['No Counts'] = {'AIN','AOU','EXTRA','FALLOFF','FALOFF','HV','ISOM','TBODY'}. Alternatively, these can be removed from the mechanism file as long as rate constants are calculated properly. 
default = set()

'Print Progress': True or False. If True, prints the stage that the algorithm is on during the reduction. Useful for identifying slow steps or errors. 
default = True


Algorithm Output: In addition to the mechanism, the algorithm will give several outputs relevant to the reduction. The output is formatted as a dictionary with the following keys and values:

'Reduced Mechanism': The reduced mechanism in the amore Mechanism class format. This format needs to be converted to the desired format externally from the mechanism. We have built in conversions for isoprene to F0AM format and camphene/gecko mechanisms to GECKO-A format. 

'Remaining Species': A list of remaining species.

'Removed Species': A list of removed species.

'Removal Order': The order in which species were set to be removed (includes the order for species that weren't removed as well). 

'Full graphs': A list of the full mechanism graph for each condition. 

'Full Yields': A list of the yields of each species from the full mechanism for each condition.

'Average full yield': The yield of each species from the full mechanism averaged across every condition. 

'Reduced Graphs': A list of the reduced graph for each condition.

'Reduced Yields': A list of the yields of each species from the reduced mechanism for each condition. 

'Tiers': The tiers in which each species was removed. Ordered list from first removed to last. This is distinct from the order. The order decides which species will be removed given the number of desired species in the mechanism. The tiers determine the actual order in which those species were removed, which is optimized to reduce the number of new edges created during the reduction process. 

'Strongly Connected Components': A list of strongly connected components in the full mechanism. 

'Reduced Strongly Connected Components': A list of strongly connected components in the reduced mechanism. 

'Groups': The species that were grouped together in the mechanism. 

'Species Dictionary': dictionary mapping each species to its index (indices are used within the algorithm for efficiency). 

'Species': A list of the species in the mechanism including newly created category species. 




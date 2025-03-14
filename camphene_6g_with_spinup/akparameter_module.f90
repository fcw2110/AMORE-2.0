!akparameter_module file for mechanism camphene_6g
      MODULE akparameter_module
                               
      IMPLICIT NONE            
                               
! max length of species names (=lco+2)
      INTEGER,PARAMETER :: maxlsp = 8
! max length of a printed reaction
      INTEGER,PARAMETER :: maxreac_char =  90
! max number of species
      INTEGER,PARAMETER :: maxsp = 129825
! max number of reactants in a reaction
      INTEGER,PARAMETER :: mxleft = 2
! max number of products in a reaction
      INTEGER,PARAMETER :: mxright = 4
! max number of reactions
      INTEGER,PARAMETER :: maxre = 775070
! max number of 3-body reactions 
      INTEGER,PARAMETER :: max_m = 6
! max number of fall-off reactions 
      INTEGER,PARAMETER :: maxfo = 39
! max number of reactions with "HV"
      INTEGER,PARAMETER :: maxhv = 43312
! max number of reactions with "CVAR"
      INTEGER,PARAMETER :: maxcvar = 1
! max number of reactions with "EXTRA"
      INTEGER,PARAMETER :: maxextra = 878
! max number of reactions with "OXYGEN"
      INTEGER,PARAMETER :: maxo2 = 2058
! max number of ISOMERIZATIONS
      INTEGER,PARAMETER :: maxiso = 7249
! max number of species undergoing phase equilibrium
      INTEGER,PARAMETER :: maxt = 26151
! max number of different types of auxiliary information
      INTEGER,PARAMETER :: maxaux = 11
! max number of different classes of RO2
      INTEGER,PARAMETER :: maxro2 = 9
! max number of PEROx in a class
      INTEGER,PARAMETER :: mxro2cl = 14161
! max number of reactions with PEROx
      INTEGER,PARAMETER :: mxrpero = 42534
! max number of different classes of dimer
      INTEGER,PARAMETER :: maxdimer = 4
! max number of dimers in a given class
      INTEGER,PARAMETER :: mxrdimer = 1
! max number of variable coefficients in CVAR type reaction
      INTEGER,PARAMETER :: maxcoe = 30
! max number of data set in function of temp. in CVAR type reaction
      INTEGER,PARAMETER :: nset = 4
! max number of angles in "HV" function
      INTEGER,PARAMETER :: maxang = 12
! -------------------------- DATA FOR CHROMOPHORE ---------------
! max # of different types of chromophore
      INTEGER,PARAMETER :: mchromo = 125
! number of chromophores  to be stored in "most used"  chromophore
!     (the "top" tables)
      INTEGER,PARAMETER :: mtopchromo = 10
! max # of species that can be stored in "most used"  chromophore
!     (the "top" tables)
      INTEGER,PARAMETER :: msptopchromo = 43312
! number of chromophores  to be stored in "regularly used"  chromophore
!     (the "med" tables)
      INTEGER,PARAMETER :: mmedchromo = 500
! max # of species that can be stored in "regularly used"  chromophore
!     (the "med" tables)
      INTEGER,PARAMETER :: mspmedchromo = 43312
! -------------------------- 
! max coefficient for interpolation of the photolytic frequencies
      INTEGER,PARAMETER :: nlo = maxang*3
! max number of boxes in the model
      INTEGER,PARAMETER :: mbox = 2
! max number of data to compute mixing height
      INTEGER,PARAMETER :: mhd = 60
! max number of isurface types
      INTEGER,PARAMETER :: msur = 4
! max number of counting species for which stoe. coff. need to be
!     evaluated in CVAR application
      INTEGER,PARAMETER :: mopc = 4
! max number of positions used to evaluate stoe. coff. for counting
!     species from the operator species
      INTEGER,PARAMETER :: mpos = 6
! max number of self reactions
      INTEGER,PARAMETER :: mself = 892
! -------------------------- 
! max number of species for which kOH is evaluated
      INTEGER,PARAMETER :: mxkOH = 15640
! max number of species for which kO3 is evaluated
      INTEGER,PARAMETER :: mxkO3 = 0
! max number of species for which kNO3 is evaluated
      INTEGER,PARAMETER :: mxkNO3 = 45416
! max number of species for which Psat is evaluated
      INTEGER,PARAMETER :: mxsat = 26152
! max number of species for which Vdep is evaluated
      INTEGER,PARAMETER :: mxdep = 26151
! -------------------------- 
! max number of precursors
      INTEGER,PARAMETER :: mxprec = 1
! max number of emitted species
      INTEGER,PARAMETER :: mes = 60
      INTEGER,PARAMETER :: maxem = 60
! max number of emitted times
      INTEGER,PARAMETER :: mtim = 50
! max number of spp assessing prod & loss rates
      INTEGER,PARAMETER :: mtr = 100
! max number of species that can be constrained
      INTEGER,PARAMETER :: maxconst = 20
! max number of lines for datapoints for constrained species
      INTEGER,PARAMETER :: maxinput = 1500
! max number of hydration reactions 
      INTEGER,PARAMETER :: maxhyd = 0
! max number of acid-base reactions 
      INTEGER,PARAMETER :: maxacid = 0
! max number of aqueous phase reactions 
      INTEGER,PARAMETER :: mxohaq = 169
! max number of aqueous mass transfer reactions 
      INTEGER,PARAMETER :: maxtr = 0
! -------------------------- 
! structure for external data used to constrain
! species emissions, concentrations, deposition...
! with possibility for using an input file
      TYPE species_data 
        LOGICAL              :: activefg 
        CHARACTER(LEN=maxlsp):: name 
        CHARACTER(LEN=10)    :: unit 
        INTEGER              :: index,npoints 
        REAL                 :: table(maxinput,2) 
      END TYPE 
!structure for storing surface data (emissions for now
! TODO later: add deposition data...
      TYPE surface_data 
        INTEGER              :: nemis 
        TYPE(species_data)   :: emission(maxem) 
      END TYPE 
                               
      END MODULE akparameter_module

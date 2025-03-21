SpeciesToAdd = {'CH3CO3';
 'CH3OO';
 'CO';
 'CO2';
 'GLYC';
 'GLYX';
 'H2O';
 'HAC';
 'HCHO';
 'HCOOH';
 'HMML';
 'HNO3';
 'HO2';
 'HOCH2CO3';
 'IHN';
 'ISOP';
 'ISOP1OH23O4OHt';
 'ISOP1OH2OO';
 'ISOP1OH2OOH';
 'ISOP1OHc';
 'ISOP1OHt';
 'ISOP3OO4OH';
 'ISOP4OHc';
 'ISOP4OHt';
 'ISOPN';
 'MACR';
 'MACR1OO';
 'MGLY';
 'MGLYOX';
 'MPAN';
 'MVK';
 'MVK3OH4OO';
 'MVK3OO4OH';
 'NO';
 'NO2';
 'NO3';
 'O2';
 'O3';
 'OH';
 'PYRAC';
 'TETRA'};
RO2ToAdd = {};
AddSpecies

i=i+1;
Rnames{i} = 'ISOP + OH = ISOP1OHc + ISOP1OHt + ISOP4OHc + ISOP4OHt';
k(:,i) = 2.7E-11.*exp(390./T).*0.63.*0.5.*3.1746031746031744;
Gstr{i,1} = 'ISOP'; Gstr{i,2} = 'OH'; 
fISOP(i)=fISOP(i)-1; fOH(i)=fOH(i)-1; fISOP1OHc(i)=fISOP1OHc(i)+0.31500000000000006; fISOP1OHt(i)=fISOP1OHt(i)+0.31500000000000006; fISOP4OHc(i)=fISOP4OHc(i)+0.25899999999999995; fISOP4OHt(i)=fISOP4OHt(i)+0.11099999999999997; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + NO = NO2 + MVK + HO2 + HCHO + IHN + CO + CH3CO3 + OH + GLYC + MGLY + CO2';
k(:,i) = F0AM_isop_ALK(T,M,2.7E-12,350.,1.190,6.,1.,0.).*1.0786181953544651;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'NO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fNO(i)=fNO(i)-1; fNO2(i)=fNO2(i)+0.8581153172379743; fMVK(i)=fMVK(i)+0.8393459139850188; fHO2(i)=fHO2(i)+0.8595220151043509; fHCHO(i)=fHCHO(i)+0.8406812668986197; fIHN(i)=fIHN(i)+0.14248800323599; fCO(i)=fCO(i)+0.023099061829346523; fCH3CO3(i)=fCH3CO3(i)+0.0127465631029664; fOH(i)=fOH(i)+0.027396261876708558; fGLYC(i)=fGLYC(i)+0.007762634626870825; fMGLY(i)=fMGLY(i)+0.0017529313760389434; fCO2(i)=fCO2(i)+0.0038349380707793103; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH + NO = NO2 + MACR + HO2 + HCHO + IHN + CO + HAC + OH + MGLY + GLYX + CO2 + CH3CO3';
k(:,i) = F0AM_isop_ALK(T,M,2.7E-12,350.,1.297,6.,1.,0.).*0.6646367673871207;
Gstr{i,1} = 'ISOP3OO4OH'; Gstr{i,2} = 'NO'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fNO(i)=fNO(i)-1; fNO2(i)=fNO2(i)+0.8685839269842226; fMACR(i)=fMACR(i)+0.8346309493016312; fHO2(i)=fHO2(i)+0.8680673708424086; fHCHO(i)=fHCHO(i)+0.8346309493016312; fIHN(i)=fIHN(i)+0.1322840079493417; fCO(i)=fCO(i)+0.06056330878638559; fHAC(i)=fHAC(i)+0.02588601592443502; fOH(i)=fOH(i)+0.055783809882286414; fMGLY(i)=fMGLY(i)+0.0039030411860723987; fGLYX(i)=fGLYX(i)+0.002006615706110695; fCO2(i)=fCO2(i)+0.004681330603840854; fCH3CO3(i)=fCH3CO3(i)+0.0032215166355148727; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + ISOP1OH2OO = MVK + HO2 + HCHO + CO + CH3CO3 + OH + GLYC + MGLY + CO2 + HAC';
k(:,i) = 6.92E-14.*0.9238741751527721;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'ISOP1OH2OO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fMVK(i)=fMVK(i)+0.9963434152685625; fHO2(i)=fHO2(i)+1.0123445283034394; fHCHO(i)=fHCHO(i)+0.9974048386382641; fCO(i)=fCO(i)+0.0311689386397219; fCH3CO3(i)=fCH3CO3(i)+0.011894303319607228; fOH(i)=fOH(i)+0.034379984700629806; fGLYC(i)=fGLYC(i)+0.006609670547477783; fMGLY(i)=fMGLY(i)+0.00375084779631845; fCO2(i)=fCO2(i)+0.0060669329341533785; fHAC(i)=fHAC(i)+0.0036229614649905717; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH + ISOP3OO4OH = MACR + HO2 + HCHO + CO + HAC + OH + NO2 + ISOPN + TETRA + CH3CO3';
k(:,i) = 5.74E-12.*0.8.*0.6939542606524177;
Gstr{i,1} = 'ISOP3OO4OH'; Gstr{i,2} = 'ISOP3OO4OH'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fMACR(i)=fMACR(i)+0.7999536983701899; fHO2(i)=fHO2(i)+1.1674890833044644; fHCHO(i)=fHCHO(i)+0.8909776220072089; fCO(i)=fCO(i)+0.21069923707157695; fHAC(i)=fHAC(i)+0.23340636757139901; fOH(i)=fOH(i)+0.1481326740901545; fNO2(i)=fNO2(i)+0.11163030054743074; fISOPN(i)=fISOPN(i)+0.0029247664952532565; fTETRA(i)=fTETRA(i)+0.1383292748934497; fCH3CO3(i)=fCH3CO3(i)+0.02552701760367252; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + ISOP3OO4OH = MVK + MACR + HO2 + HCHO + OH + CO2 + MGLY + CO + HAC + GLYX + NO2 + TETRA + CH3CO3';
k(:,i) = 3.08E-12.*0.9.*0.5597502284080398;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'ISOP3OO4OH'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fMVK(i)=fMVK(i)+0.8999449876489772; fMACR(i)=fMACR(i)+0.8991207313876696; fHO2(i)=fHO2(i)+1.0722396729384098; fHCHO(i)=fHCHO(i)+0.9227272317709176; fOH(i)=fOH(i)+0.22242941060151586; fCO2(i)=fCO2(i)+0.04992006639998454; fMGLY(i)=fMGLY(i)+0.03640958956811084; fCO(i)=fCO(i)+0.21684866169800673; fHAC(i)=fHAC(i)+0.10828249318290109; fGLYX(i)=fGLYX(i)+0.001081955685598322; fNO2(i)=fNO2(i)+0.02856899645468833; fTETRA(i)=fTETRA(i)+0.03457474334171384; fCH3CO3(i)=fCH3CO3(i)+0.020904841899171495; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH + ISOP1OH2OO = MACR + HO2 + HCHO + CO + CH3CO3 + OH + GLYC + MGLY + GLYX + CO2 + NO2 + TETRA + ISOPN + HAC';
k(:,i) = 3.94E-12.*0.705.*0.55.*0.017187467274305673;
Gstr{i,1} = 'ISOP3OO4OH'; Gstr{i,2} = 'ISOP1OH2OO'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fMACR(i)=fMACR(i)+0.7056344980367116; fHO2(i)=fHO2(i)+1.6798621636353173; fHCHO(i)=fHCHO(i)+0.8223088496998263; fCO(i)=fCO(i)+1.5326075543255717; fCH3CO3(i)=fCH3CO3(i)+0.5661879724872322; fOH(i)=fOH(i)+1.6412728207691702; fGLYC(i)=fGLYC(i)+0.3092995872684616; fMGLY(i)=fMGLY(i)+0.1603303046205561; fGLYX(i)=fGLYX(i)+0.030403134642923216; fCO2(i)=fCO2(i)+0.2641451967239775; fNO2(i)=fNO2(i)+0.1094412182660095; fTETRA(i)=fTETRA(i)+0.1024594841612495; fISOPN(i)=fISOPN(i)+0.0029703972378829708; fHAC(i)=fHAC(i)+0.31739835045675635; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + CH3OO = MVK + HO2 + HCHO + OH + CO2 + MGLY + CO + HAC + GLYX + NO2 + CH3CO3 + GLYC';
k(:,i) = 2.00E-12.*0.5.*1.8423041347935118;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'CH3OO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fCH3OO(i)=fCH3OO(i)-1; fMVK(i)=fMVK(i)+0.4920774620228843; fHO2(i)=fHO2(i)+1.399546827433573; fHCHO(i)=fHCHO(i)+1.4927728777681186; fOH(i)=fOH(i)+0.931455664961899; fCO2(i)=fCO2(i)+0.24969157502835782; fMGLY(i)=fMGLY(i)+0.18185063606055898; fCO(i)=fCO(i)+0.8226489369422602; fHAC(i)=fHAC(i)+0.2470989491206626; fGLYX(i)=fGLYX(i)+0.0051809030139157745; fNO2(i)=fNO2(i)+0.00363553349700542; fCH3CO3(i)=fCH3CO3(i)+0.07707724822773931; fGLYC(i)=fGLYC(i)+0.003467442861317008; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH + CH3OO = MACR + HO2 + HCHO + CO + HAC + OH + NO2 + ISOPN + TETRA + CH3CO3 + CO2 + MGLY + GLYX';
k(:,i) = 2.00E-12.*0.5.*1.1485834129971129;
Gstr{i,1} = 'ISOP3OO4OH'; Gstr{i,2} = 'CH3OO'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fCH3OO(i)=fCH3OO(i)-1; fMACR(i)=fMACR(i)+0.4831125644080911; fHO2(i)=fHO2(i)+1.4578098995050524; fHCHO(i)=fHCHO(i)+1.3515001895276928; fCO(i)=fCO(i)+0.31176598066293437; fHAC(i)=fHAC(i)+0.3029648752243621; fOH(i)=fOH(i)+0.2372757884930402; fNO2(i)=fNO2(i)+0.1353805323747975; fISOPN(i)=fISOPN(i)+0.0035441210867036745; fTETRA(i)=fTETRA(i)+0.1671026801418893; fCH3CO3(i)=fCH3CO3(i)+0.034852576211066376; fCO2(i)=fCO2(i)+0.010881716281151449; fMGLY(i)=fMGLY(i)+0.008203177354897903; fGLYX(i)=fGLYX(i)+0.0012126167749768341; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + HO2 = MVK + OH + HO2 + HCHO';
k(:,i) = 2.12E-13.*exp(1300./T).*0.063.*0.906545017822157;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'HO2'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fHO2(i)=fHO2(i)-1; fMVK(i)=fMVK(i)+1.0; fOH(i)=fOH(i)+1.0; fHO2(i)=fHO2(i)+1.0; fHCHO(i)=fHCHO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH + HO2 = O2 + OH + MACR + HCHO + NO2 + GLYC + TETRA + ISOPN + CH3CO3 + MGLY + ISOP1OH23O4OHt + HAC + CO + CO2 + HO2';
k(:,i) = 2.12E-13.*exp(1300./T).*0.937.*0.6130427929245701;
Gstr{i,1} = 'ISOP3OO4OH'; Gstr{i,2} = 'HO2'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fHO2(i)=fHO2(i)-1; fO2(i)=fO2(i)+0.9391416540288704; fOH(i)=fOH(i)+1.0189971761508292; fMACR(i)=fMACR(i)+0.07305548191728872; fHCHO(i)=fHCHO(i)+0.07950155252968086; fNO2(i)=fNO2(i)+0.012808798585635982; fGLYC(i)=fGLYC(i)+0.007875390610720739; fTETRA(i)=fTETRA(i)+0.15752790508513623; fISOPN(i)=fISOPN(i)+0.0013033158563720759; fCH3CO3(i)=fCH3CO3(i)+0.008425916351805039; fMGLY(i)=fMGLY(i)+0.003914867934711085; fISOP1OH23O4OHt(i)=fISOP1OH23O4OHt(i)+0.7365816566744035; fHAC(i)=fHAC(i)+0.017246464429934712; fCO(i)=fCO(i)+0.03189068596191397; fCO2(i)=fCO2(i)+0.003773772119290611; fHO2(i)=fHO2(i)+0.06085834597112959; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO = HO2 + OH + CO + CH3CO3 + NO2 + GLYC + MVK3OO4OH + GLYX + MGLY + HCHO + TETRA + CO2 + HCOOH + MVK';
k(:,i) = F0AM_isop_TUN(T,M,3.03E15,12200.,1.0E8).*0.006864765965583328;
Gstr{i,1} = 'ISOP1OH2OO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fHO2(i)=fHO2(i)+0.31908766449978604; fOH(i)=fOH(i)+1.5078686543764523; fCO(i)=fCO(i)+0.6574578695893747; fCH3CO3(i)=fCH3CO3(i)+0.1715459634573283; fNO2(i)=fNO2(i)+0.061319994678854864; fGLYC(i)=fGLYC(i)+0.0014622038791539831; fMVK3OO4OH(i)=fMVK3OO4OH(i)+0.010169633032964575; fGLYX(i)=fGLYX(i)+0.09787331846091282; fMGLY(i)=fMGLY(i)+0.22501934192199713; fHCHO(i)=fHCHO(i)+0.6091305850848808; fTETRA(i)=fTETRA(i)+0.03360894179690377; fCO2(i)=fCO2(i)+0.05342975944872746; fHCOOH(i)=fHCOOH(i)+0.01542025426148049; fMVK(i)=fMVK(i)+0.41432435526762723; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH = HO2 + OH + CO + GLYX + CH3CO3 + MGLY + PYRAC + CO2 + TETRA + HCHO + HAC + NO2 + MVK3OO4OH + MACR';
k(:,i) = F0AM_isop_TUN(T,M,1.33E9,7160.,1.0E8).*0.002516935793343671;
Gstr{i,1} = 'ISOP3OO4OH'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fHO2(i)=fHO2(i)+0.745648343870077; fOH(i)=fOH(i)+1.6604927476208116; fCO(i)=fCO(i)+1.0216529658664641; fGLYX(i)=fGLYX(i)+0.03622648115354975; fCH3CO3(i)=fCH3CO3(i)+0.09884033067037497; fMGLY(i)=fMGLY(i)+0.37717067115185016; fPYRAC(i)=fPYRAC(i)+0.05278257027490542; fCO2(i)=fCO2(i)+0.11348134316377187; fTETRA(i)=fTETRA(i)+0.08181147211249576; fHCHO(i)=fHCHO(i)+0.347135546099019; fHAC(i)=fHAC(i)+0.003714964899649836; fNO2(i)=fNO2(i)+0.09902843985386704; fMVK3OO4OH(i)=fMVK3OO4OH(i)+0.039176226111168444; fMACR(i)=fMACR(i)+0.18657795182293796; 


i=i+1;
Rnames{i} = 'MVK + OH = MVK3OO4OH';
k(:,i) = 2.6E-12.*exp(610./T).*0.75.*1.3333333333333335;
Gstr{i,1} = 'MVK'; Gstr{i,2} = 'OH'; 
fMVK(i)=fMVK(i)-1; fOH(i)=fOH(i)-1; fMVK3OO4OH(i)=fMVK3OO4OH(i)+0.9999999999999999; 


i=i+1;
Rnames{i} = 'MVK3OO4OH + HO2 = CH3CO3 + GLYC + OH + HO2 + CO + HCHO + MGLY + GLYX + CO2 + HAC';
k(:,i) = 2.12E-13.*exp(1300./T).*0.48.*1.7183943501304826;
Gstr{i,1} = 'MVK3OO4OH'; Gstr{i,2} = 'HO2'; 
fMVK3OO4OH(i)=fMVK3OO4OH(i)-1; fHO2(i)=fHO2(i)-1; fCH3CO3(i)=fCH3CO3(i)+0.7420376917659969; fGLYC(i)=fGLYC(i)+0.446150009839109; fOH(i)=fOH(i)+1.0519952764892877; fHO2(i)=fHO2(i)+0.30684602560217444; fCO(i)=fCO(i)+0.30407617632412914; fHCHO(i)=fHCHO(i)+0.22700713503387765; fMGLY(i)=fMGLY(i)+0.20515709000456453; fGLYX(i)=fGLYX(i)+0.04306762016977304; fCO2(i)=fCO2(i)+0.09661027747319864; fHAC(i)=fHAC(i)+0.008190217245527591; 


i=i+1;
Rnames{i} = 'MVK3OO4OH + NO = CH3CO3 + GLYC + NO2 + MGLY + HCHO + HO2 + CO2 + CO + HAC';
k(:,i) = F0AM_isop_ALK(T,M,2.7E-12,350.,6.161,6.,1.,0.).*0.8514928821799569;
Gstr{i,1} = 'MVK3OO4OH'; Gstr{i,2} = 'NO'; 
fMVK3OO4OH(i)=fMVK3OO4OH(i)-1; fNO(i)=fNO(i)-1; fCH3CO3(i)=fCH3CO3(i)+0.747857396911598; fGLYC(i)=fGLYC(i)+0.7358218424448306; fNO2(i)=fNO2(i)+1.0017040941759696; fMGLY(i)=fMGLY(i)+0.230606800813624; fHCHO(i)=fHCHO(i)+0.24258262268659522; fHO2(i)=fHO2(i)+0.2552408342767827; fCO2(i)=fCO2(i)+0.005949577240591924; fCO(i)=fCO(i)+0.01374997010992943; fHAC(i)=fHAC(i)+0.007756516986877983; 


i=i+1;
Rnames{i} = 'MACR + OH = MVK3OO4OH + MACR1OO';
k(:,i) = 8.0E-12.*exp(380./T).*0.53.*1.9016976262891332;
Gstr{i,1} = 'MACR'; Gstr{i,2} = 'OH'; 
fMACR(i)=fMACR(i)-1; fOH(i)=fOH(i)-1; fMVK3OO4OH(i)=fMVK3OO4OH(i)+0.5456891961744642; fMACR1OO(i)=fMACR1OO(i)+0.4543108038255357; 


i=i+1;
Rnames{i} = 'MACR1OO + HO2 = OH + CO2 + CH3OO + HCHO + CH3CO3 + HOCH2CO3 + HNO3 + CO + O3';
k(:,i) = 3.14E-12.*exp(580./T).*0.50.*1.9999999999999998;
Gstr{i,1} = 'MACR1OO'; Gstr{i,2} = 'HO2'; 
fMACR1OO(i)=fMACR1OO(i)-1; fHO2(i)=fHO2(i)-1; fOH(i)=fOH(i)+0.7466666666666667; fCO2(i)=fCO2(i)+1.1215; fCH3OO(i)=fCH3OO(i)+0.2448333333333333; fHCHO(i)=fHCHO(i)+0.42977110264975654; fCH3CO3(i)=fCH3CO3(i)+0.1318333333333333; fHOCH2CO3(i)=fHOCH2CO3(i)+0.5000000000000001; fHNO3(i)=fHNO3(i)+0.00468868256182811; fCO(i)=fCO(i)+0.4468955640169102; fO3(i)=fO3(i)+0.13; 


i=i+1;
Rnames{i} = 'MACR1OO + NO = CO2 + NO2 + HO2 + HOCH2CO3 + HCHO + HNO3 + CO';
k(:,i) = 8.7E-12.*exp(290./T).*1.0;
Gstr{i,1} = 'MACR1OO'; Gstr{i,2} = 'NO'; 
fMACR1OO(i)=fMACR1OO(i)-1; fNO(i)=fNO(i)-1; fCO2(i)=fCO2(i)+1.0; fNO2(i)=fNO2(i)+1.0; fHO2(i)=fHO2(i)+1.4008574956875655; fHOCH2CO3(i)=fHOCH2CO3(i)+1.0; fHCHO(i)=fHCHO(i)+0.10620887196617985; fHNO3(i)=fHNO3(i)+0.009377365123656218; fCO(i)=fCO(i)+0.8937911280338202; 


i=i+1;
Rnames{i} = 'MVK3OO4OH = HAC + CO + OH + HO2 + CH3CO3 + HCHO + MGLY';
k(:,i) = 2.9E7.*exp(-5297./T).*0.007647876497350468;
Gstr{i,1} = 'MVK3OO4OH'; 
fMVK3OO4OH(i)=fMVK3OO4OH(i)-1; fHAC(i)=fHAC(i)+0.9628527900686075; fCO(i)=fCO(i)+1.0; fOH(i)=fOH(i)+1.0; fHO2(i)=fHO2(i)+0.03714720993139248; fCH3CO3(i)=fCH3CO3(i)+0.011496582145164106; fHCHO(i)=fHCHO(i)+0.011496582145164106; fMGLY(i)=fMGLY(i)+0.025650627786228373; 


i=i+1;
Rnames{i} = 'MPAN + OH = HAC + CO + HMML + NO3 + NO2 + CO2';
k(:,i) = 2.9E-11.*1.0;
Gstr{i,1} = 'MPAN'; Gstr{i,2} = 'OH'; 
fMPAN(i)=fMPAN(i)-1; fOH(i)=fOH(i)-1; fHAC(i)=fHAC(i)+0.2668248364027777; fCO(i)=fCO(i)+0.2443917211973998; fHMML(i)=fHMML(i)+0.7331751635921993; fNO3(i)=fNO3(i)+0.9995380570116913; fNO2(i)=fNO2(i)+0.019945345494322152; fCO2(i)=fCO2(i)+0.022433115205377895; 


i=i+1;
Rnames{i} = 'ISOP1OH2OOH + OH = ISOP1OH23O4OHt + HAC + NO2 + TETRA + GLYC + HCHO + HO2 + ISOPN + CO + GLYX + MVK + CH3CO3 + MGLY + CO2 + OH';
k(:,i) = 1.7E-11.*exp(390./T).*0.95.*1.0751345772800869;
Gstr{i,1} = 'ISOP1OH2OOH'; Gstr{i,2} = 'OH'; 
fISOP1OH2OOH(i)=fISOP1OH2OOH(i)-1; fOH(i)=fOH(i)-1; fISOP1OH23O4OHt(i)=fISOP1OH23O4OHt(i)+0.7956634948262225; fHAC(i)=fHAC(i)+0.0290277516765331; fNO2(i)=fNO2(i)+0.016097226070770746; fTETRA(i)=fTETRA(i)+0.14937453084478236; fGLYC(i)=fGLYC(i)+0.0044636945561152335; fHCHO(i)=fHCHO(i)+0.019830476170847964; fHO2(i)=fHO2(i)+0.046513466139755134; fISOPN(i)=fISOPN(i)+0.0023520270112985815; fCO(i)=fCO(i)+0.05612215936732264; fGLYX(i)=fGLYX(i)+0.005951243266944651; fMVK(i)=fMVK(i)+0.0059805136068559125; fCH3CO3(i)=fCH3CO3(i)+0.00174236412272008; fMGLY(i)=fMGLY(i)+0.0149479222303481; fCO2(i)=fCO2(i)+0.0027259439818052332; fOH(i)=fOH(i)+0.03148686171775955; 


i=i+1;
Rnames{i} = 'ISOP1OH2OOH = MVK + HCHO + HO2 + OH + CO + HAC + MGLY + GLYX + CO2 + CH3CO3';
k(:,i) = SUN.*6.5E-6.*0.9616462832086519;
Gstr{i,1} = 'ISOP1OH2OOH'; 
fISOP1OH2OOH(i)=fISOP1OH2OOH(i)-1; fMVK(i)=fMVK(i)+0.9837862055318478; fHCHO(i)=fHCHO(i)+0.9837862055318478; fHO2(i)=fHO2(i)+1.015949520909632; fOH(i)=fOH(i)+1.0210366748038397; fCO(i)=fCO(i)+0.027340591266784436; fHAC(i)=fHAC(i)+0.007793056424747201; fMGLY(i)=fMGLY(i)+0.004309261104788184; fGLYX(i)=fGLYX(i)+0.0014521068078289876; fCO2(i)=fCO2(i)+0.005168553168685877; fCH3CO3(i)=fCH3CO3(i)+0.003904484111251355; 


i=i+1;
Rnames{i} = 'ISOP1OH23O4OHt + OH = H2O + HO2 + TETRA + MGLY + CO + GLYX + HAC + CO2 + HCHO + NO2 + CH3CO3 + GLYC';
k(:,i) = 0.67.*3.75E-11.*exp(-400./T).*1.492537313432836;
Gstr{i,1} = 'ISOP1OH23O4OHt'; Gstr{i,2} = 'OH'; 
fISOP1OH23O4OHt(i)=fISOP1OH23O4OHt(i)-1; fOH(i)=fOH(i)-1; fH2O(i)=fH2O(i)+0.19002126807666192; fHO2(i)=fHO2(i)+1.5092961995692853; fTETRA(i)=fTETRA(i)+0.20451756883579592; fMGLY(i)=fMGLY(i)+0.20265886811629924; fCO(i)=fCO(i)+1.3572107558825255; fGLYX(i)=fGLYX(i)+0.0721040575798419; fHAC(i)=fHAC(i)+0.04976601877729123; fCO2(i)=fCO2(i)+0.2883443145837762; fHCHO(i)=fHCHO(i)+0.19638658715123236; fNO2(i)=fNO2(i)+0.027735654718298346; fCH3CO3(i)=fCH3CO3(i)+0.5421640314509351; fGLYC(i)=fGLYC(i)+0.017277043428281746; 


i=i+1;
Rnames{i} = 'IHN + OH = ISOP1OH23O4OHt + HAC + HO2 + NO2 + ISOPN + GLYC + HCHO + O3 + CO2 + CH3CO3 + CO + MGLY + HNO3 + TETRA';
k(:,i) = 0.75.*8.4E-12.*exp(390./T).*0.49788835822053346;
Gstr{i,1} = 'IHN'; Gstr{i,2} = 'OH'; 
fIHN(i)=fIHN(i)-1; fOH(i)=fOH(i)-1; fISOP1OH23O4OHt(i)=fISOP1OH23O4OHt(i)+0.11312507136956017; fHAC(i)=fHAC(i)+0.16259186753802315; fHO2(i)=fHO2(i)+0.6777263185707226; fNO2(i)=fNO2(i)+0.9036428575433471; fISOPN(i)=fISOPN(i)+0.5526105288866046; fGLYC(i)=fGLYC(i)+0.15416805131577743; fHCHO(i)=fHCHO(i)+0.22298210029722973; fO3(i)=fO3(i)+0.001920530387166572; fCO2(i)=fCO2(i)+0.10270453293485886; fCH3CO3(i)=fCH3CO3(i)+0.11492584696139822; fCO(i)=fCO(i)+0.1057584104406769; fMGLY(i)=fMGLY(i)+0.03154176530084641; fHNO3(i)=fHNO3(i)+0.0012512705985429377; fTETRA(i)=fTETRA(i)+0.002954423773487596; 


i=i+1;
Rnames{i} = 'ISOP + O3 = MACR + HCHO + MVK + OH + HO2 + CO2 + CO + CH3OO';
k(:,i) = 1.1E-14.*exp(-2000./T).*0.41.*3.480487804878049;
Gstr{i,1} = 'ISOP'; Gstr{i,2} = 'O3'; 
fISOP(i)=fISOP(i)-1; fO3(i)=fO3(i)-1; fMACR(i)=fMACR(i)+0.28731604765241764; fHCHO(i)=fHCHO(i)+0.7736510161177295; fMVK(i)=fMVK(i)+0.11913104414856344; fOH(i)=fOH(i)+0.1962158374211633; fHO2(i)=fHO2(i)+0.11212333566923618; fCO2(i)=fCO2(i)+0.28521373510861947; fCO(i)=fCO(i)+0.28521373510861947; fCH3OO(i)=fCH3OO(i)+0.28521373510861947; 


i=i+1;
Rnames{i} = 'ISOP + NO3 = MVK + HCHO + OH + HO2 + NO2 + H2O + IHN + CH3CO3 + MGLY + GLYX + ISOPN + CO + GLYC + MACR + CO2 + CH3OO + HAC + TETRA';
k(:,i) = 2.95E-12.*exp(-450./T).*0.45.*2.2222222222222223;
Gstr{i,1} = 'ISOP'; Gstr{i,2} = 'NO3'; 
fISOP(i)=fISOP(i)-1; fNO3(i)=fNO3(i)-1; fMVK(i)=fMVK(i)+0.3750651542037578; fHCHO(i)=fHCHO(i)+0.6751153561959872; fOH(i)=fOH(i)+0.28287862123610025; fHO2(i)=fHO2(i)+0.33552385328751233; fNO2(i)=fNO2(i)+1.2316384920704873; fH2O(i)=fH2O(i)+0.014536916796088786; fIHN(i)=fIHN(i)+0.22519268589756658; fCH3CO3(i)=fCH3CO3(i)+0.10982531850538074; fMGLY(i)=fMGLY(i)+0.06240178664353517; fGLYX(i)=fGLYX(i)+0.01161476980673359; fISOPN(i)=fISOPN(i)+0.12465263066680043; fCO(i)=fCO(i)+0.025650379389485594; fGLYC(i)=fGLYC(i)+0.05017229519342039; fMACR(i)=fMACR(i)+0.03985331563146081; fCO2(i)=fCO2(i)+0.1270713792148816; fCH3OO(i)=fCH3OO(i)+0.004332081466402648; fHAC(i)=fHAC(i)+0.03868215886274038; fTETRA(i)=fTETRA(i)+0.007580022972815465; 


i=i+1;
Rnames{i} = 'ISOPN + OH = HO2 + HCHO + NO2 + GLYX + CH3CO3 + CO + HCOOH + MGLY + GLYC + CO2 + HAC + O3 + PYRAC + OH';
k(:,i) = 1.0E-11.*0.3122693293003057;
Gstr{i,1} = 'ISOPN'; Gstr{i,2} = 'OH'; 
fISOPN(i)=fISOPN(i)-1; fOH(i)=fOH(i)-1; fHO2(i)=fHO2(i)+0.717642317371835; fHCHO(i)=fHCHO(i)+0.4893898670367368; fNO2(i)=fNO2(i)+1.1590713723410329; fGLYX(i)=fGLYX(i)+0.005647235274332208; fCH3CO3(i)=fCH3CO3(i)+0.6434013526702688; fCO(i)=fCO(i)+1.6308019791098085; fHCOOH(i)=fHCOOH(i)+0.014503905688553273; fMGLY(i)=fMGLY(i)+0.05390090132608592; fGLYC(i)=fGLYC(i)+0.04273751312862674; fCO2(i)=fCO2(i)+0.08103725685623206; fHAC(i)=fHAC(i)+0.2605738579777859; fO3(i)=fO3(i)+0.0012726979182080891; fPYRAC(i)=fPYRAC(i)+0.015555588678387553; fOH(i)=fOH(i)+0.031821237193476515; 


i=i+1;
Rnames{i} = 'IHN = OH + NO2 + CO + HO2 + CH3CO3 + GLYX + MGLY + CO2 + PYRAC + HCOOH + GLYC + HCHO + MVK3OO4OH + HAC + MACR1OO';
k(:,i) = 0.455.*0.55.*SUN.*4.0E-4.*0.04407913213234748;
Gstr{i,1} = 'IHN'; 
fIHN(i)=fIHN(i)-1; fOH(i)=fOH(i)+1.4176705457792933; fNO2(i)=fNO2(i)+1.3039966301365848; fCO(i)=fCO(i)+1.5124067538819757; fHO2(i)=fHO2(i)+0.7314112963244811; fCH3CO3(i)=fCH3CO3(i)+0.1564456464337696; fGLYX(i)=fGLYX(i)+0.1399733863640397; fMGLY(i)=fMGLY(i)+0.37002868467554983; fCO2(i)=fCO2(i)+0.038300753248998294; fPYRAC(i)=fPYRAC(i)+0.25053961653261025; fHCOOH(i)=fHCOOH(i)+0.10909244657014404; fGLYC(i)=fGLYC(i)+0.010396781391206707; fHCHO(i)=fHCHO(i)+0.014178752734666981; fMVK3OO4OH(i)=fMVK3OO4OH(i)+0.25770167962592877; fHAC(i)=fHAC(i)+0.017653699132477354; fMACR1OO(i)=fMACR1OO(i)+0.009598767314538105; 


i=i+1;
Rnames{i} = 'TETRA + OH = OH + CO + HCHO + MGLY + HAC + HO2 + CH3CO3 + GLYX + CO2 + GLYC';
k(:,i) = 9.85E-12.*exp(410./T).*0.06556261186677513;
Gstr{i,1} = 'TETRA'; Gstr{i,2} = 'OH'; 
fTETRA(i)=fTETRA(i)-1; fOH(i)=fOH(i)-1; fOH(i)=fOH(i)+0.9863090478980518; fCO(i)=fCO(i)+1.340557834448424; fHCHO(i)=fHCHO(i)+0.413590969980788; fMGLY(i)=fMGLY(i)+0.3455946213264275; fHAC(i)=fHAC(i)+0.5123922137324055; fHO2(i)=fHO2(i)+0.18910226706595865; fCH3CO3(i)=fCH3CO3(i)+0.1420071796131566; fGLYX(i)=fGLYX(i)+0.005816699436681304; fCO2(i)=fCO2(i)+0.31176323654888494; fGLYC(i)=fGLYC(i)+0.01009424713580615; 


i=i+1;
Rnames{i} = 'HMML + OH = MGLY + OH + CH3CO3 + HCOOH';
k(:,i) = 4.33E-12.*1.0;
Gstr{i,1} = 'HMML'; Gstr{i,2} = 'OH'; 
fHMML(i)=fHMML(i)-1; fOH(i)=fOH(i)-1; fMGLY(i)=fMGLY(i)+0.7; fOH(i)=fOH(i)+0.7; fCH3CO3(i)=fCH3CO3(i)+0.3; fHCOOH(i)=fHCOOH(i)+0.3; 


i=i+1;
Rnames{i} = 'PYRAC + OH = CH3CO3 + CO2';
k(:,i) = 8.0E-13.*1.0;
Gstr{i,1} = 'PYRAC'; Gstr{i,2} = 'OH'; 
fPYRAC(i)=fPYRAC(i)-1; fOH(i)=fOH(i)-1; fCH3CO3(i)=fCH3CO3(i)+1.0; fCO2(i)=fCO2(i)+1.0; 


i=i+1;
Rnames{i} = 'PYRAC = CH3CO3 + CO2 + HO2';
k(:,i) = SUN.*3.1E-3.*1.0;
Gstr{i,1} = 'PYRAC'; 
fPYRAC(i)=fPYRAC(i)-1; fCH3CO3(i)=fCH3CO3(i)+1.0; fCO2(i)=fCO2(i)+1.0; fHO2(i)=fHO2(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOPN = MGLY + OH + NO2 + GLYC + CO + HO2 + HAC + CH3CO3 + HCHO + TETRA + GLYX + CO2';
k(:,i) = SUN.*3.0E-5.*0.3797678129083134;
Gstr{i,1} = 'ISOPN'; 
fISOPN(i)=fISOPN(i)-1; fMGLY(i)=fMGLY(i)+0.23770798689968142; fOH(i)=fOH(i)+0.930028919234801; fNO2(i)=fNO2(i)+1.1459313814289989; fGLYC(i)=fGLYC(i)+0.34936184995681996; fCO(i)=fCO(i)+1.2054243008060916; fHO2(i)=fHO2(i)+1.036676750489003; fHAC(i)=fHAC(i)+0.4417003975501733; fCH3CO3(i)=fCH3CO3(i)+0.3130835924185361; fHCHO(i)=fHCHO(i)+0.23686725784963863; fTETRA(i)=fTETRA(i)+0.0018389259609572087; fGLYX(i)=fGLYX(i)+0.014535777129257388; fCO2(i)=fCO2(i)+0.02835563261433705; 


i=i+1;
Rnames{i} = 'TETRA = CO + HO2 + OH + HAC + CH3CO3 + HCHO + GLYX + MGLY + CO2 + GLYC';
k(:,i) = SUN.*6.49E-6.*0.29437312298812907;
Gstr{i,1} = 'TETRA'; 
fTETRA(i)=fTETRA(i)-1; fCO(i)=fCO(i)+0.16278856901369929; fHO2(i)=fHO2(i)+1.104239779210889; fOH(i)=fOH(i)+1.0333140370833862; fHAC(i)=fHAC(i)+0.5257570521316967; fCH3CO3(i)=fCH3CO3(i)+0.04500874875878681; fHCHO(i)=fHCHO(i)+0.0039191503403688925; fGLYX(i)=fGLYX(i)+0.5366196648583186; fMGLY(i)=fMGLY(i)+0.42786104164719196; fCO2(i)=fCO2(i)+0.03331403708338644; fGLYC(i)=fGLYC(i)+0.3835471097052147; 


i=i+1;
Rnames{i} = 'HOCH2CO3 + HO2 = HO2 + HCHO + OH + O3';
k(:,i) = KAPHO2.*0.44.*2.272727272727273;
Gstr{i,1} = 'HOCH2CO3'; Gstr{i,2} = 'HO2'; 
fHOCH2CO3(i)=fHOCH2CO3(i)-1; fHO2(i)=fHO2(i)-1; fHO2(i)=fHO2(i)+0.43999999999999995; fHCHO(i)=fHCHO(i)+0.9999999999999996; fOH(i)=fOH(i)+0.8499999999999996; fO3(i)=fO3(i)+0.14999999999999997; 


i=i+1;
Rnames{i} = 'HOCH2CO3 + NO = NO2 + HO2 + HCHO';
k(:,i) = KAPNO.*1.0;
Gstr{i,1} = 'HOCH2CO3'; Gstr{i,2} = 'NO'; 
fHOCH2CO3(i)=fHOCH2CO3(i)-1; fNO(i)=fNO(i)-1; fNO2(i)=fNO2(i)+1.0; fHO2(i)=fHO2(i)+1.0; fHCHO(i)=fHCHO(i)+1.0; 


i=i+1;
Rnames{i} = 'HOCH2CO3 + NO2 = HCHO + CO';
k(:,i) = KFPAN.*1.0;
Gstr{i,1} = 'HOCH2CO3'; Gstr{i,2} = 'NO2'; 
fHOCH2CO3(i)=fHOCH2CO3(i)-1; fNO2(i)=fNO2(i)-1; fHCHO(i)=fHCHO(i)+0.9999999999999992; fCO(i)=fCO(i)+0.9999999999999992; 


i=i+1;
Rnames{i} = 'HOCH2CO3 + NO3 = NO2 + HO2 + HCHO';
k(:,i) = KRO2NO3.*1.74.*1.0;
Gstr{i,1} = 'HOCH2CO3'; Gstr{i,2} = 'NO3'; 
fHOCH2CO3(i)=fHOCH2CO3(i)-1; fNO3(i)=fNO3(i)-1; fNO2(i)=fNO2(i)+1.0; fHO2(i)=fHO2(i)+1.0; fHCHO(i)=fHCHO(i)+1.0; 


i=i+1;
Rnames{i} = 'HOCH2CO3 = HCHO + HO2';
k(:,i) = 1.00e-11.*0.7.*0.1.*1.4285714285714286;
Gstr{i,1} = 'HOCH2CO3'; 
fHOCH2CO3(i)=fHOCH2CO3(i)-1; fHCHO(i)=fHCHO(i)+1.0; fHO2(i)=fHO2(i)+1.0; 


i=i+1;
Rnames{i} = 'HAC + OH = MGLYOX + HO2';
k(:,i) = 1.6E-12.*exp(305/T).*1.0;
Gstr{i,1} = 'HAC'; Gstr{i,2} = 'OH'; 
fHAC(i)=fHAC(i)-1; fOH(i)=fOH(i)-1; fMGLYOX(i)=fMGLYOX(i)+1.0; fHO2(i)=fHO2(i)+1.0; 


i=i+1;
Rnames{i} = 'HAC = CH3CO3 + HCHO + HO2';
k(:,i) = J22.*1.0;
Gstr{i,1} = 'HAC'; 
fHAC(i)=fHAC(i)-1; fCH3CO3(i)=fCH3CO3(i)+1.0; fHCHO(i)=fHCHO(i)+1.0; fHO2(i)=fHO2(i)+1.0; 


i=i+1;
Rnames{i} = 'MACR1OO + NO2 = MPAN';
k(:,i) = F0AM_isop_TROE2(T,M,2.591E-28,0.,-6.87,1.125E-11,0.,-1.105,0.3).*1;
Gstr{i,1} = 'MACR1OO'; Gstr{i,2} = 'NO2'; 
fMACR1OO(i)=fMACR1OO(i)-1; fNO2(i)=fNO2(i)-1; fMPAN(i)=fMPAN(i)+1.0; 


i=i+1;
Rnames{i} = 'MPAN = MACR1OO + NO2';
k(:,i) = 1.58E16.*exp(-13500./T).*1;
Gstr{i,1} = 'MPAN'; 
fMPAN(i)=fMPAN(i)-1; fMACR1OO(i)=fMACR1OO(i)+1.0; fNO2(i)=fNO2(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OHt + O2 = ISOP1OH2OO';
k(:,i) = 3.6E-13.*1.0;
Gstr{i,1} = 'ISOP1OHt'; Gstr{i,2} = 'O2'; 
fISOP1OHt(i)=fISOP1OHt(i)-1; fO2(i)=fO2(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OHt + O2 = ISOP1OH2OO';
k(:,i) = 7.5E-13.*1.0;
Gstr{i,1} = 'ISOP1OHt'; Gstr{i,2} = 'O2'; 
fISOP1OHt(i)=fISOP1OHt(i)-1; fO2(i)=fO2(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OHc + O2 = ISOP1OH2OO';
k(:,i) = 7.5E-13.*1.0;
Gstr{i,1} = 'ISOP1OHc'; Gstr{i,2} = 'O2'; 
fISOP1OHc(i)=fISOP1OHc(i)-1; fO2(i)=fO2(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OHc + O2 = ISOP1OH2OO';
k(:,i) = 1.4E-13.*0.9999999999999999;
Gstr{i,1} = 'ISOP1OHc'; Gstr{i,2} = 'O2'; 
fISOP1OHc(i)=fISOP1OHc(i)-1; fO2(i)=fO2(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP4OHt + O2 = ISOP3OO4OH';
k(:,i) = 4.9E-13.*1.0;
Gstr{i,1} = 'ISOP4OHt'; Gstr{i,2} = 'O2'; 
fISOP4OHt(i)=fISOP4OHt(i)-1; fO2(i)=fO2(i)-1; fISOP3OO4OH(i)=fISOP3OO4OH(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP4OHt + O2 = ISOP3OO4OH';
k(:,i) = 6.5E-13.*1.0;
Gstr{i,1} = 'ISOP4OHt'; Gstr{i,2} = 'O2'; 
fISOP4OHt(i)=fISOP4OHt(i)-1; fO2(i)=fO2(i)-1; fISOP3OO4OH(i)=fISOP3OO4OH(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP4OHc + O2 = ISOP3OO4OH';
k(:,i) = 6.5E-13.*1.0;
Gstr{i,1} = 'ISOP4OHc'; Gstr{i,2} = 'O2'; 
fISOP4OHc(i)=fISOP4OHc(i)-1; fO2(i)=fO2(i)-1; fISOP3OO4OH(i)=fISOP3OO4OH(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP4OHc + O2 = ISOP3OO4OH';
k(:,i) = 2.1E-13.*1.0;
Gstr{i,1} = 'ISOP4OHc'; Gstr{i,2} = 'O2'; 
fISOP4OHc(i)=fISOP4OHc(i)-1; fO2(i)=fO2(i)-1; fISOP3OO4OH(i)=fISOP3OO4OH(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO = ISOP1OHt';
k(:,i) = 1.83E14.*exp(-8930./T).*0.02118426799387381;
Gstr{i,1} = 'ISOP1OH2OO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fISOP1OHt(i)=fISOP1OHt(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO = ISOP1OHt';
k(:,i) = 2.22E15.*exp(-10355./T).*0.9074619345439865;
Gstr{i,1} = 'ISOP1OH2OO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fISOP1OHt(i)=fISOP1OHt(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO = ISOP1OHc';
k(:,i) = 2.24E15.*exp(-10865./T).*0.9065374669922348;
Gstr{i,1} = 'ISOP1OH2OO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fISOP1OHc(i)=fISOP1OHc(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO = ISOP1OHc';
k(:,i) = 1.79E14.*exp(-8830./T).*0.0019314252209139552;
Gstr{i,1} = 'ISOP1OH2OO'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fISOP1OHc(i)=fISOP1OHc(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH = ISOP4OHt';
k(:,i) = 2.08E14.*exp(-9400./T).*0.02153885631686045;
Gstr{i,1} = 'ISOP3OO4OH'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fISOP4OHt(i)=fISOP4OHt(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH = ISOP4OHt';
k(:,i) = 2.49E15.*exp(-10890./T).*0.5556107333986585;
Gstr{i,1} = 'ISOP3OO4OH'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fISOP4OHt(i)=fISOP4OHt(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH = ISOP4OHc';
k(:,i) = 2.49E15.*exp(-11112./T).*0.554809959575464;
Gstr{i,1} = 'ISOP3OO4OH'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fISOP4OHc(i)=fISOP4OHc(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP3OO4OH = ISOP4OHc';
k(:,i) = 1.75E14.*exp(-9054./T).*0.0008941012159487325;
Gstr{i,1} = 'ISOP3OO4OH'; 
fISOP3OO4OH(i)=fISOP3OO4OH(i)-1; fISOP4OHc(i)=fISOP4OHc(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OOH + OH = ISOP1OH2OO';
k(:,i) = 2.0E-12.*exp(200./T).*0.0008287992518882422;
Gstr{i,1} = 'ISOP1OH2OOH'; Gstr{i,2} = 'OH'; 
fISOP1OH2OOH(i)=fISOP1OH2OOH(i)-1; fOH(i)=fOH(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OOH + OH = ISOP1OH2OO';
k(:,i) = 2.0E-12.*exp(200./T).*0.014804513801957072;
Gstr{i,1} = 'ISOP1OH2OOH'; Gstr{i,2} = 'OH'; 
fISOP1OH2OOH(i)=fISOP1OH2OOH(i)-1; fOH(i)=fOH(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OOH + OH = ISOP1OH2OO';
k(:,i) = 4.6E-12.*exp(200./T).*0.946092085605398;
Gstr{i,1} = 'ISOP1OH2OOH'; Gstr{i,2} = 'OH'; 
fISOP1OH2OOH(i)=fISOP1OH2OOH(i)-1; fOH(i)=fOH(i)-1; fISOP1OH2OO(i)=fISOP1OH2OO(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + HO2 = ISOP1OH2OOH + O2';
k(:,i) = 2.12E-13.*exp(1300./T).*0.937.*0.9068840330263218;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'HO2'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fHO2(i)=fHO2(i)-1; fISOP1OH2OOH(i)=fISOP1OH2OOH(i)+1.0; fO2(i)=fO2(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + HO2 = O2 + ISOP1OH2OOH';
k(:,i) = 2.12E-13.*exp(1300./T).*0.0030717569489127814;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'HO2'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fHO2(i)=fHO2(i)-1; fO2(i)=fO2(i)+1.0; fISOP1OH2OOH(i)=fISOP1OH2OOH(i)+1.0; 


i=i+1;
Rnames{i} = 'ISOP1OH2OO + HO2 = O2 + ISOP1OH2OOH';
k(:,i) = 2.12E-13.*exp(1300./T).*0.01219670702636547;
Gstr{i,1} = 'ISOP1OH2OO'; Gstr{i,2} = 'HO2'; 
fISOP1OH2OO(i)=fISOP1OH2OO(i)-1; fHO2(i)=fHO2(i)-1; fO2(i)=fO2(i)+1.0; fISOP1OH2OOH(i)=fISOP1OH2OOH(i)+1.0; 


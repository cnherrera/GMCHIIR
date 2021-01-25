#
# Main script to extract the information of each molecular cloud (GMC) and HII region
# Scripts reads the molecular cloud and HII region catalogs, for each GMC the scripts find the closest
# HII region. A visualisation of the correlation between the properties of the paired GMCs and HII region is done.
#


import os 
import math
import sys
from pdb import set_trace as stop
import numpy as np

import astropy.units as u
from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import constants as ct
from astropy.utils.data import get_pkg_data_filename

import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf

from sklearn.linear_model import LinearRegression
import seaborn as sns
from typing import List

import pdb

np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=True)

#===================================================================================

def checkdist(xgalhii,ygalhii,xgalgmc,ygalgmc,sizehii,radgmc,distance):
    """
    Measure the distance between the GMCs and HII regions

    Parameters:
    ----------
    xgalhii : Position in the galaxy the X axis of the HII region in arcsec
    ygalhii : Position in the galaxy the Y axis of the HII region in arcsec
    xgalgmc : Position in the galaxy the X axis of the molecular cloud in arcsec
    ygalgmc : Position in the galaxy the Y axis of the molecular cloud in arcsec
    sizehii : Size of the HII region in pc
    radgmc : Size of the molecular cloud in pc
    distance : Distance of the galaxy in kpc

    Returns:
    ---------
    mindist : minimum distance between a GMC and HII region in parsec
    inddist : index for that minimum distance
    idovergmc: index for the GMCs
    idoverhii: index for the HII regions
    idgmcalone : index for all unpaired GMCs 
    idhiialone : index for all unpaired HII regions
    
    """

    dists  = np.zeros((2,len(xgalgmc)))
    # For each GMC we look for the closes HII region
    for j in range(len(xgalgmc)):
        distas = ((xgalgmc[j]-xgalhii)**2 + (ygalgmc[j]-ygalhii)**2)**0.5
        dist = np.radians(distas/3600)*distance*1e6  #distance between the GMC and all HII regions in pc
        # Save the index and value of the minimum distance for a given GMC
        dists[0,j] = int(np.argmin(dist))
        dists[1,j] = min(dist)
    mindist = dists[1,] ; inddist = dists[0,]

    idgmc = [] ; idhii = [] ; distmin = []
    indall = range(len(inddist))  # it will be the index of my index position of the HIIR.

    # Removing HIIR that are doubled, i.e. the same HIIR paired with more than 1 GMC.
    for idint, it in enumerate(inddist):  #  a for loop in all gmcs, reading the index of the HIIR
        indw = np.where(inddist == it)[0]  # Looking for the same index of HIIR, "it", in the entire saved index.
        # If the same GMC has being paired twice, we should have several indices in indw
        if len(indw) > 1:
            igmc = np.extract(inddist == it,indall)  # extract the index of the gmcs associated to the same HIIR, igmc.
            imin = np.argmin(np.extract(inddist == it,mindist)) # get the index of the minimum distance between all index that are it
            dmin = np.min(np.extract(inddist == it,mindist)) # get the  minimum distance between all index that are it 
            indgmc  = igmc[imin] # Index of the GMC that is the closest to the HIIR that was associated to different GMCs. Only this one will be saved.
            if it not in idhii:
                idhii.append(int(it))
                distmin.append(dmin)
                idgmc.append(indgmc)
        else:
            idhii.append(int(it)) #index of the HIIRegion
            distmin.append(np.extract(inddist == it,mindist))
            idgmc.append(idint)

    # Index  idhii idgmc
    addsize = (sizehii[idhii] + radgmc[idgmc])#*4#/3.
    #tmpoverid = np.argwhere(dists[1,idgmc] < (sizehii[idhii]*2)) # HII with GMCs < 2 sizeHII type: List[int]
    tmpoverid = np.argwhere(mindist[idgmc] < addsize)
    overid = [int(item) for item in tmpoverid]
    idovergmc = [idgmc[item] for item in overid]
    idoverhii = [idhii[item] for item in overid]
    allgmc = np.arange(len(radgmc))
    allhii = np.arange(len(sizehii))
    idgmcalone = np.delete(allgmc,idovergmc).tolist()
    idhiialone = np.delete(allhii,idoverhii).tolist()
    return mindist,inddist,idovergmc,idoverhii,idgmcalone,idhiialone
    

# Write a list of sources into DS9 format
def writeds9(galnam,namegmc,rahii,dechii,pind,ragmc,decgmc,comment):
    
    f=open('../ds9tables/%s%s-HIIregions-%s.reg' % (galnam,namegmc,comment) ,"w+")
    f.write("# Region file format: DS9 version 4.1\n")
    f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
    f.write("fk5\n")
    for j in range(len(rahii)):
        f.write('circle(%12.8f,%12.8f,1") # text={%i} width=3\n' % (rahii[j], dechii[j],pind[j]))
    f.close()

    f=open('../ds9tables/%s%s-GMCs-%s.reg' % (galnam,namegmc,comment),"w+")
    f.write("# Region file format: DS9 version 4.1\n")
    f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
    f.write("fk5\n")
    for j in range(len(ragmc)):
        f.write('circle(%12.8f,%12.8f,1") # text={%i} width=3 color=red\n' % (ragmc[j], decgmc[j],j))
    f.close() 

def outnan(var1):   #Taking out the NaN values
    nonan = var1[~np.isnan(var1)]
    return nonan


#===================================================================================
# Defining paths to the files and files names
dirhii = "../../Catalogs-HIIregions/hii_region_enhanced_catalogs/"
dirgmc = "../../Catalogs-CPROPS/"
diralma = '../../ALMA-LP-delivery/delivery_v3p3/'

hiicats = [f for f in os.listdir(dirhii)]

#----------------------------------------------------------------
# Name of CPROPS catalog. Change to:
# Native resolution: "_co21_v1p0_props"
# Homogenized resolution: "_co21_120pc_props" # 7 galaxies
# Homogenized resolution: "_co21_v150pc_props" # 8 galaxies
# namegmc = "_co21_v1p0_props"#"_co21_150pc_props"
#-------------------------------------------------
# 22-jun-2020: New GMC catalogs!
# native resolution in folder: Catalogs-CPROPS/native
# homogenized: Catalogs-CPROPS/homogenized
# matched: Catalogs-CPROPS/matched

typegmc = "ST1p5/homogenized"  # ST1p5/native
dirgmc = dirgmc+typegmc+"/"
with open("name_gmc.txt","r") as myfile:
   namegmc = myfile.read().replace('\n', '')  #"_12m+7m+tp_co21_120pc_props"  # "_12m+7m+tp_co21_native_props"

#-----------------------------------------

pdf1 = fpdf.PdfPages("Histograms_all_GMC%s.pdf" % namegmc)
pdf2 = fpdf.PdfPages("Correlations_galbygal_GMC%s.pdf"  % namegmc)  # type: PdfPages
pdf3 = fpdf.PdfPages("Correlations_allgals_GMC%s.pdf" % namegmc)  # type: PdfPages

#=======================================================================================
# Defining empty vectors to save the variables of all galaxies
galaxias=[]

LumHacorr=[] ; GaldisHII=[] ; SizepcHII=[] ; sigmavHII=[] ; metaliHII=[] ; varmetHII=[]
DisHIIGMC=[] ; SizepcGMC=[] ; sigmavGMC=[] ; MasscoGMC=[] ; aviriaGMC=[] ; Sigmamole=[]
TpeakGMC =[] ; tauffGMC =[] ; numGMConHII = []

LumHacorrover=[] ; GaldisHIIover=[] ; SizepcHIIover=[] ; sigmavHIIover=[] ; metaliHIIover=[] ; varmetHIIover=[]
DisHIIGMCover=[] ; SizepcGMCover=[] ; sigmavGMCover=[] ; MasscoGMCover=[] ; aviriaGMCover=[] ; Sigmamoleover=[]
TpeakGMCover=[]  ; tauffGMCover = []

LumHacorrno=[] ; GaldisHIIno=[] ; SizepcHIIno=[] ; sigmavHIIno=[] 
SizepcGMCno=[] ; sigmavGMCno=[] ; MasscoGMCno=[] ; aviriaGMCno=[] ; Sigmamoleno=[]
TpeakGMCno =[] ; tauffGMCno =[]


RAgmcover = [] ; DECgmcover = []  
RAhiiover = [] ; DEChiiover = []
RAgmcall = [] ; DECgmcall = []
RAhiiall = [] ; DEChiiall = []
#==========================================================================
# Limits in the properties of HIIR and GMCs
#this file is a product if this script! (if the script runs for the first time, comment the following line).
xlim,ylim,xx,yy=pickle.load(open('limits_properties.pickle',"rb"))

#=============================================================================================================
# Loop in all galaxies. Do histograms and individual galaxy plots.
print "Starting loop in all galaxies [i], do histograms and individual galaxy plots"
for i in range(len(hiicats)):

    galnam = hiicats[i].split("_")[0]
    galaxias.append(galnam)

    print "-*"*20
    print "Galaxy name: %s" % galnam
    print "-------------------------"
    thii = Table.read(dirhii+hiicats[i])

    #thii.info #thii.colnames
    
    #Information of the galaxy
    PAgal   = thii['PA'][0] #PA of galaxy in degrees
    inclgal = thii['INCL'][0] #incl inclination of galaxy in degrees
    racen   =  thii['RA_CENTER'][0] ; deccen  =  thii['DEC_CENTER'][0]

    # Check HII regions with bptflag 0, i.e. OK
    flg = thii['BPTFLAG']
    idfl = (np.where(flg == 0)[0]).tolist()
    
    #Information of individual HII regions
    rahii  = thii['RA'][idfl] ; dechii = thii['DEC'][idfl]
    pind = thii['PHANGS_INDEX'][idfl]

    #lhahiicorr = thii['LHA']*10.**(0.4*np.interp(6563, [5530,6700],[1.,0.74])*3.1*thii['EBV']) # Correcting
    #lhahiicorr = thii['CLHA']  # erg/s
    #elhahiicorr = thii['CLHA_ERR'] # Error in Lhalpha extinction corrected
    sigmahii = thii['HA_SIG'][idfl] 
    metalhii = thii['METAL_SCAL'][idfl] 
    vamethii = thii['OFF_SCAL'][idfl] 
    disthii  = thii['DISTMPC'][0]

    #=============================================================
    # Corresponding CO data and GMC catalog
    #--------------------------------------------
    #data = fits.getdata(("%s%s_co21_v1p0_props.fits" % (dirgmc,galnam)), 1)
    #tgmc = Table.read(("%s%s_co21_v1p0_props.fits" % (dirgmc,galnam)))
    print "Reading table"
    tgmc = Table.read(("%s%s%s.fits" % (dirgmc,galnam,namegmc)))

    dist_gal_Mpc = tgmc['DISTANCE_PC'][0]/1e6
    print dist_gal_Mpc
    print tgmc['BEAMFWHM_PC'][0]
    
    s2n = tgmc['S2N']
    ids2n = (np.where(s2n > 0)[0]).tolist()

    sigvgmc= tgmc['SIGV_KMS'][ids2n]
    ragmc  = tgmc['XCTR_DEG'][ids2n]
    decgmc = tgmc['YCTR_DEG'][ids2n]
    fluxco = tgmc['FLUX_KKMS_PC2'][ids2n]
    radgmc = tgmc['RAD_PC'][ids2n]
    radnogmc=tgmc['RAD_NODC_NOEX'][ids2n]
    tpgmc  = tgmc['TMAX_K'][ids2n]
    if (galnam != 'ngc1672' and galnam != 'ic5332'):
       massco = tgmc['MLUM_MSUN'][ids2n] 
       avir   = tgmc['VIRPARAM'][ids2n]
       tauff  = tgmc['TFF_MYR'][ids2n]*10**6
       Sigmamol_co=tgmc['SURFDENS'][ids2n]
    else:
       massco = tgmc['FLUX_KKMS_PC2'][ids2n]*4.3/0.69
       avir  = 5.*( sigvgmc*1e5 )**2 * radgmc * ct.pc.cgs.value /massco/ct.M_sun.cgs.value/ct.G.cgs.value
       Sigmamol_co  = massco/( radgmc**2 * math.pi )
       RAD3 = (radgmc**2*50)**(0.33)
       rhogmc = 1.26 * massco / (4/3 * math.pi * RAD3**3) * ct.M_sun.cgs.value/ct.pc.cgs.value**3
       arg = 3. * math.pi / (32 * ct.G.cgs.value * rhogmc)
       tauff = [math.sqrt(f)/365/24/3600 for f in arg]
       tauff = np.array(tauff)
    
    Sigmamol_vir = tgmc['MVIR_MSUN'][ids2n] / ( radgmc**2 * math.pi )
        

    #=========================================================================================
    #Correct LHa and size HII by the new distance measurement.
    
    lhahiicorr = thii['CLHA'][idfl]*(dist_gal_Mpc/disthii)**2  # erg/s
    sizehii  = thii['SIZE'][idfl]*(dist_gal_Mpc/disthii) #pc
    
    #==========================================================================================
    # Write to DS9 readable table
    wds9 = writeds9(galnam,namegmc,rahii,dechii,pind,ragmc,decgmc,"all_regions")

    #==========================================================================================
    # Galac tic distance in HII regions and GMCs

    # HII
    center_pos = SkyCoord(racen, deccen, unit=(u.deg, u.deg), frame='fk5')
    offsets    = SkyCoord(rahii, dechii, unit=(u.deg, u.deg))
    PAs    = center_pos.position_angle(offsets).degree * math.pi / 180
    gcdist = offsets.separation(center_pos)
    Rplane = gcdist.arcsecond

    galPA  = PAs - PAgal
    xplane = Rplane * np.cos(galPA)
    yplane = Rplane * np.sin(galPA)
    xgalhii   = xplane
    ygalhii   = yplane / np.cos(inclgal)   ###  To be changed if we don't want to deproject.
    rgalhii   = (xgalhii**2 + ygalhii**2)**0.5   #arcsec
    rgalhii   = np.radians(rgalhii/3600)*dist_gal_Mpc*1e3 # kpc
    psigalhii = np.arctan2(ygalhii, xgalhii)

    # GMC
    offsets    = SkyCoord(ragmc, decgmc, unit=(u.deg, u.deg))
    PAs    = center_pos.position_angle(offsets).degree * math.pi / 180
    gcdist = offsets.separation(center_pos)
    Rplane = gcdist.arcsecond

    galPA  = PAs - PAgal
    xplane = Rplane * np.cos(galPA)
    yplane = Rplane * np.sin(galPA)
    xgalgmc   = xplane
    ygalgmc   = yplane / np.cos(inclgal) ###  To be changed if we don't want to deproject.
    rgalgmc   = (xgalgmc**2 + ygalgmc**2)**0.5   #arcsec
    psigalgmc = np.arctan2(ygalgmc, xgalgmc)

    if galnam == 'ngc1672':
        discen = rgalgmc
        print dist_gal_Mpc
        print discen[0], discen[20]
        limd = 15 #0.5/dist_gal_Mpc*180/math.pi*3600
        sigvgmc=sigvgmc[discen >limd]
        ragmc  = ragmc[discen >limd]
        decgmc = decgmc[discen >limd]
        fluxco = fluxco[discen >limd]
        radgmc = radgmc[discen >limd]
        radnogmc= radnogmc[discen >limd]
        tpgmc  = tpgmc[discen >limd]
        massco = massco[discen >limd]
        avir        = avir[discen >limd]
        Sigmamol_co = Sigmamol_co[discen >limd]
        tauff       = tauff[discen >limd]
        xgalgmc = xgalgmc[discen>limd]
        ygalgmc = ygalgmc[discen>limd]

    #==========================================================================================
    #  Distance between HII and GMCs ;  in arcsec
    # mindist: minimum distance in oparsec
    # inddist: index for that minimu distance
    # idovergmc: index for the gmcs
    # idoverhii: index for the hii regions
    mindist,inddist,idovergmc,idoverhii,idgmcno,idhiino=checkdist(xgalhii,ygalhii,xgalgmc,ygalgmc,sizehii,radgmc,dist_gal_Mpc)

    # For each HII region, I get the number and index of GMCs that are at a distance < 2*size
    #and save variables with all data becaus of the index.
    numgmcs = [np.zeros(len(xgalhii)),[None]*len(xgalhii)]
    for j in range(len(xgalhii)):
        dstas = ((xgalgmc-xgalhii[j])**2 + (ygalgmc-ygalhii[j])**2)**0.5
        dst = np.radians(dstas/3600)*dist_gal_Mpc*1e6  #dist in pc
        mylim = [dst < sizehii[j]*2]
        numgmcs[1][j] = np.where(mylim[0]==True)[0]
        numgmcs[0][j] = len(np.where(mylim[0]==True)[0])
    numGMConHII.append(numgmcs)


    # Defining individual arrays
    print ("Saving variables in external file: Clouds_HIIregions_positions_%s%s.pickle" % (galnam,namegmc))
    with open(('Clouds_HIIregions_positions_%s%s.pickle' % (galnam,namegmc)), "wb") as f:
        pickle.dump([galnam, rahii, dechii,pind,idoverhii,ragmc,decgmc,idovergmc], f)
        
    LumHacorr_galo = lhahiicorr[idoverhii]
    GaldisHII_galo = rgalhii[idoverhii]
    SizepcHII_galo = sizehii[idoverhii]
    sigmavHII_galo = sigmahii[idoverhii]
    metaliHII_galo = metalhii[idoverhii]
    varmetHII_galo = vamethii[idoverhii]

    DisHIIGMC_galo = mindist[idovergmc]
    SizepcGMC_galo = radgmc[idovergmc]
    sigmavGMC_galo = sigvgmc[idovergmc]
    MasscoGMC_galo = massco[idovergmc]/1e5
    aviriaGMC_galo = avir[idovergmc]
    Sigmamole_galo = Sigmamol_co[idovergmc]
    FluxCOGMC_galo = fluxco[idovergmc]
    TpeakGMC_galo  = tpgmc[idovergmc]
    tauffGMC_galo  = tauff[idovergmc]

    RAgmc = ragmc[idovergmc] ; DECgmc = decgmc[idovergmc]
    RAhii = rahii[idoverhii] ; DEChii = dechii[idoverhii]
    phii = pind[idoverhii]

    wds9 = writeds9(galnam,namegmc,RAhii,DEChii,phii,RAgmc,DECgmc,"overlapped")
    # Save in a single array for all galaxies -
    LumHacorr.append(lhahiicorr) ; GaldisHII.append(rgalhii) ; SizepcHII.append(sizehii) ; sigmavHII.append(sigmahii) ; metaliHII.append(metalhii) ;  varmetHII.append(vamethii)
    DisHIIGMC.append(mindist) ; SizepcGMC.append(radgmc) ;  sigmavGMC.append(sigvgmc) ; MasscoGMC.append(massco/1e5) ; aviriaGMC.append(avir)
    Sigmamole.append(Sigmamol_co) ; TpeakGMC.append(tpgmc) ; tauffGMC.append(tauff)

    LumHacorrover.append(LumHacorr_galo) ; GaldisHIIover.append(GaldisHII_galo) ; SizepcHIIover.append(SizepcHII_galo) ; sigmavHIIover.append(sigmavHII_galo) ; metaliHIIover.append(metaliHII_galo) ; varmetHIIover.append(varmetHII_galo)
    DisHIIGMCover.append(DisHIIGMC_galo) ; SizepcGMCover.append(SizepcGMC_galo) ; sigmavGMCover.append(sigmavGMC_galo) ; MasscoGMCover.append(MasscoGMC_galo) ; aviriaGMCover.append(aviriaGMC_galo) ; Sigmamoleover.append(Sigmamole_galo)
    TpeakGMCover.append(TpeakGMC_galo)   ; tauffGMCover.append(tauffGMC_galo)

    LumHacorrno.append(lhahiicorr[idhiino]) ; GaldisHIIno.append(rgalhii[idhiino]) ; SizepcHIIno.append(sizehii[idhiino]) ; sigmavHIIno.append(sigmahii[idhiino])
    SizepcGMCno.append(radgmc[idgmcno]) ;  sigmavGMCno.append(sigvgmc[idgmcno]) ; MasscoGMCno.append(massco[idgmcno]) ; aviriaGMCno.append(avir[idgmcno])
    Sigmamoleno.append(Sigmamol_co[idgmcno]) ; TpeakGMCno.append(tpgmc[idgmcno]) ; tauffGMCno.append(tauff[idgmcno])


    RAgmcover.append(RAgmc) ; DECgmcover.append(DECgmc)
    RAhiiover.append(RAhii)  ; DEChiiover.append(DEChii)

    RAgmcall.append(ragmc) ; DECgmcall.append(decgmc)
    RAhiiall.append(rahii) ; DEChiiall.append(dechii)   

    # Quantifying
    LHa_all  = np.nansum(lhahiicorr[np.isfinite(lhahiicorr)])
    LHa_galo = np.nansum(LumHacorr_galo[np.isfinite(LumHacorr_galo)])

    Lco_all  = np.nansum(fluxco)
    Lco_galo = np.nansum(FluxCOGMC_galo)

    print "Total HII regions: %i, overlapping clouds: %i"  % (len(lhahiicorr),len(LumHacorr_galo))
    print "Total GMCs %i,  overlapping HII regions: %i"  % (len(fluxco),len(FluxCOGMC_galo))
    print "Total Ha lum [erg/s]: %10.2E" %LHa_all
    print "Ha lum for those with GMCs overlapped[erg/s]: %10.2E %5.1f %%" % (LHa_galo,LHa_galo*100/LHa_all)
    print "Total CO Flux [K km/s pc2]: %10.2E" %Lco_all
    assert isinstance(Lco_all, object)
    print "CO Flux for those with HII regions overlapped[erg/s]: %10.2E %5.1f %%" % (Lco_galo,Lco_galo*100/Lco_all)
    print "-"*30
    if i == 0:
        file = open("Table1.txt","w")
        file.write(r" \multirow{2}{*}{Galaxy} & \multicolumn{2}{c}{Total}& \multicolumn{2}{c}{Overlapping} & \multicolumn{2}{c}{H$\alpha$ luminosity [erg s$^{-1}$]} & \multicolumn{2}{c}{CO flux [K km s$^{-1}$ pc$^2$]}\\"+"\n")
        file.write(r"\cline{2-9}"+"\n")
        file.write(r"\hline"+"\n")
        file.write(r" & \hii\ regions & GMCs & \% \hii\ & \% GMCs & Total & Overlapping & Total & Overlapping\\"+"\n")
        file.write(r"\hline"+"\n"+r"\noalign{\smallskip}"+"\n")
    if i != 0:
        file = open("Table1.txt","a")
    if len(LumHacorr_galo)!=0:
        file.write("%s & %i & %i & %i\\%% & %i\\%% & %10.2E & %i\\%% & %10.2E & %i\\%% \\\ \n" % (galnam,len(lhahiicorr), len(fluxco),round(len(LumHacorr_galo)*100.00/len(lhahiicorr)),round(len(LumHacorr_galo)*100/len(fluxco)),LHa_all,round(LHa_galo*100/LHa_all),Lco_all,round(Lco_galo*100/Lco_all)))
    if len(LumHacorr_galo)==0:
        file.write("%s & %i & %i & %i\\%% & %i\\%% & %10.2E & %i\\%% & %10.2E & %i\\%% \\\ \n" % (galnam,len(lhahiicorr), len(fluxco),0,0,LHa_all,0,Lco_all,0))
    file.close()

    print len(LumHacorr_galo)
    if len(LumHacorr_galo)==0:
        continue

    #==========================================================================================
    ## PLOTS
    #============
    print "Starting plots "
    title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal','verticalalignment':'bottom'} 
    marker_style = dict(markersize=3)

    # -------------------
    # Histograms
    # Taking out the NaNs values since otherwise the histograms are not nice.
    Sigmamol_co_n = outnan(Sigmamol_co)
    avir_n        = outnan(avir)
    massco_n      = outnan(massco)
    sigv_kms_n    = outnan(sigvgmc)
    rad_nodc_noex_n = outnan(radnogmc)
    dists_n         = outnan(mindist)
    sigmahii_n      = outnan(sigmahii)
    lhacorrall_n    = outnan(lhahiicorr[(np.array(lhahiicorr) < 1e50)])
    sizehii_n       = outnan(sizehii)
    rgalhii_n       = outnan(rgalhii)
    metalhii_n      = outnan(metalhii[(np.abs(metalhii) < 30)])
    vamethii_n      = outnan(vamethii[(np.abs(vamethii) < 30)])
    sigmahii_cl_n      = outnan(sigmavHII_galo)
    lhacorrall_cl_n    = outnan(LumHacorr_galo)
    sizehii_cl_n       = outnan(SizepcHII_galo)
    rgalhii_cl_n       = outnan(GaldisHII_galo)
    metalhii_cl_n      = outnan(metaliHII_galo)
    vamethii_cl_n      = outnan(varmetHII_galo)
    print "hola1"
    if galnam=='ic5332':
        continue   
    arrays = [rgalhii_n,sizehii_n,lhacorrall_n,sigmahii_n,metalhii_n,vamethii_n,dists_n,rad_nodc_noex_n,sigv_kms_n,massco_n,avir_n,Sigmamol_co_n]
    labsname = ['Galactocentric radius [kpc]','HII region size [pc]', r'Luminosity H$\alpha$ [erg/s]',r'$\sigma_{v}$ HII region [km/s]','Metallicity','Variation metallicity','Distance  HII-GMC [pc]','GMC size [pc]',r'$\sigma_v$ [km/s]',r'Mass$_{CO}$ [M$_{\odot}$]',r'$\alpha_{vir}$',r'$\Sigma_{mol}$']
    arraycl=[rgalhii_cl_n,sizehii_cl_n,lhacorrall_cl_n,sigmahii_cl_n,metalhii_cl_n,vamethii_cl_n]

    fig, axs = plt.subplots(6,2,figsize=(8,12))
    plt.subplots_adjust(hspace = 0.4)
    fig.suptitle('Galaxy %s - Histograms' %galnam.upper(), fontsize=15,va='top')
    axs = axs.ravel()      # to put in 1D the axs
    print (" Histograms saved in: Histograms_all_GMC%s.pdf" % namegmc)
    for z in range(len(arrays)):
        minv = np.min(arrays[z])
        maxv = np.max(arrays[z])
        if z < 6 and z!=2 :
            axs[z].hist(arrays[z],alpha=0.5,bins=np.arange(minv,maxv,(maxv-minv)/20),label=['All HII regions'])
            axs[z].hist(arraycl[z],alpha=0.5,bins=np.arange(minv,maxv,(maxv-minv)/20),label=['HII regions with overlapping GMCs'])
        if (z > 5) and (z != 9) and (z != 11):
            axs[z].hist(arrays[z],alpha=0.5,bins=20)
        if (z == 2) or (z == 9) or (z == 11):
            if z == 2 :
                axs[z].hist(arrays[z],alpha=0.5,bins=np.logspace(np.log10(minv),np.log10(maxv),20))
                axs[z].hist(arraycl[z],alpha=0.5,bins=np.logspace(np.log10(minv),np.log10(maxv),20))
            else:
                axs[z].hist(arrays[z],alpha=0.5,bins=np.logspace(np.log10(minv),np.log10(maxv),20))
            axs[z].set_xscale("log")
        axs[z].title.set_text(labsname[z])
    axs[1].legend(prop={'size': 6})
  
    pdf1.savefig(fig)
    plt.close()
    
    # Plot HII parameters vs GMC parameters
    title_font = {'fontname': 'Arial', 'size': '18', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    marker_style = dict(markersize=3)

    # -------------------
    arrayxax = [GaldisHII_galo,SizepcHII_galo,LumHacorr_galo,sigmavHII_galo,metaliHII_galo,varmetHII_galo]
    arrayyay = [DisHIIGMC_galo,MasscoGMC_galo,SizepcGMC_galo,Sigmamole_galo,sigmavGMC_galo,aviriaGMC_galo,TpeakGMC_galo,tauffGMC_galo]

    labsxax = ['Galactocentric radius [kpc]', 'HII region size [pc]', r'Luminosity H$\alpha$ [erg/s]',
               r'$\sigma_{v}$ HII region [km/s]','Metallicity','Metallicity variation']
    labsyay = ['Dist. HII-GMC [pc]', r'M$_{\rm CO}$ [10^5 M$_{\odot}$]', 'GMC size [pc]', r'$\Sigma_{\rm mol}$',
               r'$\sigma_{\rm v}$ [km/s]', r'$\alpha_{\rm vir}$',r'CO $T_{\rm peak}$',r'$\tau_{\rm ff}$']
        
    print "Plotting HII region vs GMC parameters for individual galaxies"
    for k in range(len(arrayxax)):
        sns.set(style="white", color_codes=True)
        fig, axs = plt.subplots(4, 2, sharex='col', figsize=(8, 10), gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('Galaxy %s' % galnam.upper(), fontsize=18, va='top')
        axs = axs.ravel()
        for i in range(len(labsyay)):
            xax = arrayxax[k]
            yay = arrayyay[i]
            xaxt = xax
            if "Metallicity" not in labsxax[k]:
                xaxt = np.log10(xax)
            yayt = np.log10(yay)
            idok = np.where((abs(yayt) < 100000) & (abs(xaxt) < 100000))
            xax = xaxt[idok] ; yay = yayt[idok]
            lim1 = np.nanmedian(xax) - np.nanstd(xax)*4
            lim2 = np.nanmedian(xax) + np.nanstd(xax)*4
            indlim = np.where((xax < lim2) & (xax>lim1))
            xax = xax[indlim] ; yay = yay[indlim]
            sns.set(color_codes=True)        
            if len(xax)>2:
                xmin = np.amin(xax)
                xmax = np.amax(xax)
                xprang = (xmax-xmin)*0.1
                x=xax.reshape((-1,1))
                y=yay
                model = LinearRegression().fit(x, y)
                r_sq = model.score(x, y)
                y_pred = model.intercept_ + model.coef_ * x.ravel()
                sns.regplot(x=xax,y=yay,ax=axs[i])
                x0,xf = xlim[k]
                y0,yf = ylim[i]
                xprang = (xf - x0) * 0.05
                yprang = (yf - y0) * 0.05
                axs[i].text(x0+xprang/2, y0+yprang ,'R sq: %6.4f'%(r_sq))
                axs[i].set(xlim=(x0,xf))
                axs[i].set(ylim=(y0,yf))
            axs[i].set(ylabel=labsyay[i])
            axs[i].grid()
        axs[6].set(xlabel=labsxax[k])
        axs[7].set(xlabel=labsxax[k])
        axs[7].set(ylim=(5.2,yf))

        pdf2.savefig(fig)
        plt.close()

pdf1.close()
pdf2.close()

#==================================================================================
# Obtaining LHa/HIIsize^2 ratio
ratlin = [None]*len(LumHacorrover)
for j in range(len(galaxias)):
    if galaxias[j] != 'ic5332':
        ratlin[j] =  ((LumHacorrover[j])/(SizepcHIIover[j]**2))
    else:
        ratlin[j] = LumHacorrover[j]

#==================================================================================
# Saving the parameters to be read by another procedure.

print "Plots of all galaxies together"
arrayxax = [GaldisHIIover,SizepcHIIover,LumHacorrover,sigmavHIIover,ratlin,metaliHIIover,varmetHIIover]
arrayyay = [DisHIIGMCover,MasscoGMCover,SizepcGMCover,Sigmamoleover,sigmavGMCover,aviriaGMCover,TpeakGMCover,tauffGMCover]
labsxax = ['log(Galactocentric radius) [kpc]','log(HII region size) [pc]', r'log(Luminosity H$\alpha$) [erg s$^{-1}$]',r'log($\sigma_{\rm v}$ HII region) [km s$^{-1}$]',r'log(Lum H$\alpha$/HII region size$^2$)','Metallicity','Metallicity variation']
labsyay = ['log(Dist. HII-GMC) [pc]',r'log(M$_{\rm CO}$) [10$^5$ M$_{\odot}$]','log(GMC size) [pc]',r'log($\Sigma_{\rm mol}$)',r'log($\sigma_{\rm v}$) [km s$^{-1}$]',r'log($\alpha_{vir}$)',r'log(CO $T_{\rm peak}$ [K])',r'log($\tau_{\rm ff}$) [yr]']

print "Saving variables in external file."
with open(('Galaxies_paired_GMC%s.pickle' % namegmc), "wb") as f:
    pickle.dump([galaxias,arrayyay,arrayxax,RAgmcover,DECgmcover,RAhiiover,DEChiiover,labsxax,labsyay],f)

print "Saving variables in external file."
with open(('Galaxies_variables_all_GMC%s.pickle' % namegmc), "wb") as f:
    pickle.dump([GaldisHII,SizepcHII,LumHacorr,sigmavHII,metaliHII,varmetHII,numGMConHII,MasscoGMC],f)

print "Saving variables in external file."
with open(('Galaxies_variables_notover_GMC%s.pickle' % namegmc), "wb") as f:
    pickle.dump([LumHacorrno,GaldisHIIno,SizepcHIIno,sigmavHIIno,SizepcGMCno,sigmavGMCno,MasscoGMCno,aviriaGMCno,Sigmamoleno,TpeakGMCno,tauffGMCno],f)






#
# Scripts to measure the limits for all variables
# in order to plot all correlations with the same scale.
#

import numpy as np
import pickle

def outinf(v1):
    """
    Function to take out all values that are Inf.
    """
    v1n = np.array(v1)
    v1n2 = v1n[np.isfinite(v1n)]
    return v1n2


#-------------------------------------------------------------------------------------------------------------
dirhii = "../../Catalogs-HIIregions/hii_region_enhanced_catalogs/"
dirgmc = "../../Catalogs-CPROPS/"
diralma = '../../ALMA-LP-delivery/delivery_v3p3/'

# The kind of data we will plot, native or homogenized data.
namegmc = "_12m+7m+tp_co21_150pc_props" # native

galaxias,GMCprop,HIIprop,RAgmc,DECgmc,RAhii,DEChii,labsxax,labsyay= pickle.load(open(('Galaxies_variables_GMC%s.pickle' % namegmc),"rb"))
GaldisHII,SizepcHII,LumHacorr,sigmavHII,metaliHII,varmetHII,numGMConHII,MasscoGMC = pickle.load(open(('Galaxies_variables_notover_GMC%s.pickle' % namegmc), "rb"))

GaldisHIIover,SizepcHIIover,LumHacorrover,sigmavHIIover,ratlin,metaliHIIover,varmetHIIover = HIIprop
DisHIIGMCover,MasscoGMCover,SizepcGMCover,Sigmamoleover,sigmavGMCover,aviriaGMCover,TpeakGMCover,tauffGMCover = GMCprop

xlim = []
ylim = []
xlimt = []
ylimt = []

for k in range(len(labsxax)):
    xaxtmp = HIIprop[k]
    xaxt = np.concatenate([f.tolist() for f in xaxtmp])
    xax = outinf(xaxt)
    if k<5:
        xax = np.log10(xax)
    xlim1 = np.nanmedian(xax) - np.nanstd(xax)*4
    xlim2 = np.nanmedian(xax) + np.nanstd(xax)*4
    xmin = np.nanmin(xax)
    xmax = np.nanmax(xax)
    xrang = xmax-xmin
    xi = xmin - xrang*0.1
    xf = xmax + xrang*0.1
    xlim1 = np.amax([xlim1,xi]) 
    xlim2 = np.amin([xlim2,xf]) 
    xlim.append([xi,xf])
    xlimt.append([xlim1,xlim2])

    
for k in range(len(labsyay)):
    yaytmp = GMCprop[k]
    yayt = np.concatenate([f.tolist() for f in yaytmp])
    yay = outinf(yayt)
    yay = np.log10(yay)
    ylim1 = np.nanmedian(yay) - np.nanstd(yay)*4
    ylim2 = np.nanmedian(yay) + np.nanstd(yay)*4    
    ymin = np.nanmin(yay)
    ymax = np.nanmax(yay)
    yrang = ymax-ymin
    yi = ymin - yrang*0.1
    yf = ymax + yrang*0.1
    ylim1 = np.amax([ylim1,yi]) 
    ylim2 = np.amin([ylim2,yf])     
    ylim.append([yi,yf])
    ylimt.append([ylim1,ylim2])
    
print ("Saving variables in external file: limits_properties.pickle")
with open('limits_properties.pickle', "wb") as f:
        pickle.dump([xlimt,ylimt,labsxax,labsyay], f)

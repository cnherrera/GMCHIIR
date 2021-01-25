#
# 2D contour plots 
#

import pickle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import matplotlib.backends.backend_pdf as fpdf
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=True)

#==============================================================================

dirhii = "../../Catalogs-HIIregions/hii_region_enhanced_catalogs/"
dirgmc = "../../Catalogs-CPROPS/"
diralma = '../../ALMA-LP-delivery/delivery_v3p3/'
#-------------------------------------------------
# 22-jun-2020: New GMC catalogs!
# native resolution in folder: Catalogs-CPROPS/native
# homogenized: Catalogs-CPROPS/homogenized
# matched: Catalogs-CPROPS/matched

typegmc = "homogenized"  # native
dirgmc = dirgmc+typegmc+"/"
namegmc = "_12m+7m+tp_co21_150pc_props"  # "_12m+7m+tp_co21_native_props"


galaxias,GMCprop,HIIprop,RAgmc,DECgmc,RAhii,DEChii,labsxax,labsyay= pickle.load(open(('Galaxies_variables_GMC%s.pickle' % namegmc),"rb"))
GaldisHII,SizepcHII,LumHacorr,sigmavHII,metaliHII,varmetHII,numGMConHII,MasscoGMC = pickle.load(open(('Galaxies_variables_notover_GMC%s.pickle' % namegmc), "rb"))
arrayyay = GMCprop
arrayxax = HIIprop
GaldisHIIover,SizepcHIIover,LumHacorrover,sigmavHIIover,ratlin,metaliHIIover,varmetHIIover = HIIprop
shortlab = ['HIIGMCdist', 'Mco','GMCsize','Smol', 'sigmav','avir','TpeakCO','tauff']

# Limits in the properties of HIIR and GMCs
xlim,ylim,xx,yy=pickle.load(open('limits_properties.pickle',"rb"))


# #=============================================================================================================================
# #-------------------------------
# # Working with the color maps. https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# #
print "Plots of all galaxies together -  2D contour"
print '-' *30
#"defining color maps"
spring = cm.get_cmap('spring',256)
mspring    = ListedColormap(spring(np.linspace(0.1, 0.4, 256)))
mycmp=['Greys','Purples','Blues','Greens','Oranges','Reds',mspring,'Wistia']
#
# # Getting color maps, with 25 values only
mgrey = cm.get_cmap('Greys',25)
mPurples = cm.get_cmap('Purples',25)
mBlues = cm.get_cmap('Blues',25)
mGreens = cm.get_cmap('Greens',25)
mOranges = cm.get_cmap('Oranges',25)
mReds = cm.get_cmap('Reds',25)
mpink = cm.get_cmap('pink',25)
mWistia = cm.get_cmap('Wistia',25)
 # Extracting only some colors from the 25 color maps. This is to use them as contours to not start from black.
mygrey    = ListedColormap(mgrey(np.linspace(0.75, 0.95, 25)))
myPurples = ListedColormap(mPurples(np.linspace(0.75, 0.95, 25))) 
myBlues   = ListedColormap(mBlues(np.linspace(0.75, 0.95, 25))) 
myGreens  = ListedColormap(mGreens(np.linspace(0.75, 0.95, 25))) 
myOranges = ListedColormap(mOranges(np.linspace(0.75, 0.95, 25))) 
myReds    = ListedColormap(mReds(np.linspace(0.75, 0.95, 25)))                      
mypink    = ListedColormap(mpink(np.linspace(0.75, 0.95, 25))) 
myWistia  = ListedColormap(mWistia(np.linspace(0.75, 0.95, 25)))

mycon = [mygrey,myPurples,myBlues,myGreens,myOranges,myReds,mspring,myWistia]
mycol=['grey','purple','blue','green','orange','red', 'pink', 'orange']
# #--------------------------------------------------------------------
#
pdf = fpdf.PdfPages("Correlations_allgals_2Dcontours_GMC%s.pdf" % namegmc)  # type: PdfPages
#
# # Changing the transparency value of each galaxy.
myalpha=np.arange(0.9,0.3,-0.08)
xticks5 = [8.3,8.4,8.5, 8.6, 8.7]
nlevs = 3

# print "Starting for"
for k in range(5): # not plotting metallicity
    fig, axs = plt.subplots(4, 2,sharex='col',figsize=(8,10),gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace = 0.3)
    fig.suptitle('All galaxies - Overlapping HIIregions and GMCs', fontsize=15,va='top')
    axs = axs.ravel()
    for i in range(len(labsyay)):
        for j in range(len(galaxias)):
            xaxt = np.log10(arrayxax[k][j])
            yayt = np.log10(arrayyay[i][j])
            idok = np.where(yayt == yayt)
            xax = xaxt[idok] ; yay = yayt[idok]
            if galaxias[j]=='ngc2835':
                print xax,yay
            if len(xax)>4:
                sns.kdeplot(xax,yay,ax=axs[i],shade=True,shade_lowest=False,alpha=myalpha[j],cmap=mycmp[j],n_levels=nlevs,)
                sns.kdeplot(xax,yay,ax=axs[i],linewidths=0.5, cmap=mycon[j], color=mycol[j],label='%s' % galaxias[j], legend=True,n_levels=nlevs)
        axs[i].set(ylabel=labsyay[i])
        axs[i].grid()
        x0,xf = xlim[k]
        y0,yf = ylim[i]
        axs[i].set(xlim=(x0,xf))
        axs[i].set(ylim=(y0,yf))
    axs[0].legend(prop={'size': 8})
    axs[6].set(xlabel=labsxax[k])
    axs[7].set(xlabel=labsxax[k])
    pdf.savefig(fig)
    plt.close()

pdf.close()
#
# #=========================


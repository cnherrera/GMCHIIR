#
# Plot correlations between properties of paired GMC and HII regions.
# Measure linear regression between properties.
#

import sys
import numpy as np
import math
import pickle
from astropy import constants as ct
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
import seaborn as sns
from typing import List

np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=True)

#===================================================================================

def checknaninf(v1,v2,lim1,lim2):
    """
    From 2 arrays, take out values that are Inf in either 2 lists.
    """
    v1n = np.array(v1)
    v2n = np.array(v2)
    indok = np.where((np.absolute(v1n) < lim1) & (np.absolute(v2n) < lim2))[0].tolist()
    nv1n = v1n[indok].tolist()
    nv2n = v2n[indok].tolist()
    return nv1n,nv2n

def bindata(xaxall,yayall,mybin):
    """
    Bin X-Y data 
    """
    xran	= np.amax(xaxall)-np.amin(xaxall)
    xspa	= xran/mybin
    xsta	= np.amin(xaxall)+xspa/2
    xfin	= np.amax(xaxall)-xspa/2
    xbinned = np.linspace(xsta,xfin, mybin)
    ybinned = []
    eybinned = []
    nybinned = []
    for t in range(mybin):
       idxs = np.where(abs(xaxall-xbinned[t])<xspa/2)
       yayin = yayall[idxs]
       nyayin = len(yayin)
       myayin = np.nanmean(yayin)
       syayin = np.nanstd(yayin)
       ybinned.append(myayin)
       eybinned.append(syayin)
       nybinned.append(nyayin)
    return xbinned,ybinned,eybinned,nybinned


#-=============================================================
dirhii = "../../Catalogs-HIIregions/hii_region_enhanced_catalogs/"
dirgmc = "../../Catalogs-CPROPS/"
diralma = '../../ALMA-LP-delivery/delivery_v3p3/'

# The kind of data we will plot, native or homogenized data.
namegmc = "_12m+7m+tp_co21_150pc_props" # native

#=====================================================================================
# Reading properties of clouds

galaxias,GMCprop,HIIprop,RAgmc,DECgmc,RAhii,DEChii,labsxax,labsyay= pickle.load(open(('Galaxies_variables_GMC%s.pickle' % namegmc),"rb"))
GaldisHII,SizepcHII,LumHacorr,sigmavHII,metaliHII,varmetHII,numGMConHII,MasscoGMC = pickle.load(open(('Galaxies_variables_notover_GMC%s.pickle' % namegmc), "rb"))
arrayyay = GMCprop
arrayxax = HIIprop
GaldisHIIover,SizepcHIIover,LumHacorrover,sigmavHIIover,ratlin,metaliHIIover,varmetHIIover = HIIprop
DisHIIGMCover,MasscoGMCover,SizepcGMCover,Sigmamoleover,sigmavGMCover,aviriaGMCover,TpeakGMCover,tauffGMCover = GMCprop
shortlab = ['HIIGMCdist', 'Mco','GMCsize','Smol', 'sigmav','avir','TpeakCO','tauff']
MassesCO = [1e5*i for i in MasscoGMCover] #

# Limits in the properties of HIIR and GMCs
xlim,ylim,xx,yy=pickle.load(open('limits_properties.pickle',"rb"))

#===============================================================
# Plots of correlations with dots for each pair
print "Plots of all galaxies together"

df = sns.load_dataset('iris')

marker_style = dict(markersize=4)
#xticks1 = np.arange(8.4,9,0.2)
xticks5 = [8.3,8.4,8.5, 8.6, 8.7]


pdf3 = fpdf.PdfPages("Correlations_allgals_GMC%s.pdf" % namegmc)  # type: PdfPages
print "Starting loop to create figures of all galaxies together - points"
for k in range(len(arrayxax)):
    sns.set(style='white',color_codes=True)
    fig, axs = plt.subplots(4, 2,sharex='col',figsize=(9,10),dpi=80,gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace = 0.3)
    fig.suptitle('All galaxies - Overlapping HIIregions and GMCs', fontsize=18,va='top')
    axs = axs.ravel()
    # Galactic distance vs: Mco, avir, sigmav,Sigmamol
    for i in range(len(labsyay)):
        for j in range(len(galaxias)):
            xax2 = [h for h in arrayxax[k][j]]
            yay2 = [h for h in arrayyay[i][j]]
            if k < 5:
                xax2 = np.log10(xax2)
            yay2 = np.log10(yay2)
            axs[i].plot(xax2, yay2,'8',label='%s'%galaxias[j],alpha=0.7, **marker_style)
        axs[i].set(ylabel=labsyay[i])
        axs[i].grid()
        yaytmp = arrayyay[i]
        xaxtmp = arrayxax[k]
        xaxall = np.concatenate([f.tolist() for f in xaxtmp])
        yayall = np.concatenate([f.tolist() for f in yaytmp])
        if k<5:
            xaxall = np.log10(xaxall)
        yayall = np.log10(yayall)
        idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
        xaxall = xaxall[idok] ; yayall = yayall[idok]
        lim1=np.nanmedian(xaxall)-np.nanstd(xaxall)*4
        lim2=np.nanmedian(xaxall)+np.nanstd(xaxall)*4
        #pdb.set_trace()
        indlim = np.where((xaxall < lim2) & (xaxall>lim1))
        xaxall = xaxall[indlim] ; yayall = yayall[indlim]
        #pdb.set_trace()
        xmin = np.amin(xaxall)
        xmax = np.amax(xaxall)
        #pdb.set_trace()
        xprang = (xmax - xmin) * 0.1
        x = xaxall.reshape((-1, 1))
        y = yayall
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        y_pred = model.intercept_ + model.coef_ * x.ravel()
        axs[i].plot(xaxall, y_pred,'-')
        #sn.regplot(x=xaxall, y=yayall, ax=axs[i])
        x0 = xmin+xprang
        #x0, xf = axs[i].get_xlim()
        #y0, yf = axs[i].get_ylim()
        x0,xf = xlim[k]
        y0,yf = ylim[i]
        axs[i].text(x0, y0, 'R sq: %6.4f' % (r_sq))
#        axs[i].set(xlim=(xmin - xprang, xmax + xprang))
        axs[i].set(xlim=(x0,xf))
        axs[i].set(ylim=(y0,yf))
        axs[0].legend(prop={'size': 5})
    axs[6].set(xlabel=labsxax[k])
    axs[7].set(xlabel=labsxax[k])
    pdf3.savefig(fig)
    plt.close()

pdf3.close()

#==============================================
# Plot binned 
pdf4 = fpdf.PdfPages("Correlations_allgals_GMC_binned%s.pdf" % namegmc)  # type: PdfPages
print "Starting loop to create figures of all galaxies together - binned"
for k in range(len(arrayxax)):
#    print "starting for k"
    sns.set(style='white',color_codes=True)
    fig, axs = plt.subplots(4, 2,sharex='col',figsize=(8,10),gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace = 0.3)
    fig.suptitle('All galaxies - Overlapping HIIregions and GMCs', fontsize=18,va='top')
    axs = axs.ravel()
#    print "starting for i"
    # Galactic distance vs: Mco, avir, sigmav,Sigmamol
    for i in range(len(labsyay)):
        yaytmp = arrayyay[i] ; xaxtmp = arrayxax[k]
        xaxall = np.concatenate([f.tolist() for f in xaxtmp])
        yayall = np.concatenate([f.tolist() for f in yaytmp])
        yayall = np.log10(yayall)
        if "Metallicity" in labsxax[k]:
            xaxall = xaxall
        else:
            xaxall = np.log10(xaxall)
        idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
        xaxall = xaxall[idok] ; yayall = yayall[idok]
        lim1=np.nanmedian(xaxall)-np.nanstd(xaxall)*4
        lim2=np.nanmedian(xaxall)+np.nanstd(xaxall)*4
        indlim = np.where((xaxall < lim2) & (xaxall>lim1))
        xaxall = xaxall[indlim] ; yayall = yayall[indlim]
        mybin = 20
        xbinned,ybinned,eybinned,nybinned=bindata(xaxall,yayall,mybin)
        # if there is any nan inside
        ido = np.where(np.array(nybinned) != 0)
        xbinned  = xbinned[ido]
        ybinned  = [g for g in np.array(ybinned)[ido]]
        eybinned = [g for g in np.array(eybinned)[ido]]
        nybinned = [g for g in np.array(nybinned)[ido]]
        # Plot binned data
        mysize = np.array(nybinned).astype(float)
        mysize = (mysize-np.min(mysize))/(np.max(mysize)-np.min(mysize))*9+3
        mylims = [np.argmin(mysize),np.argmax(mysize)]
        mylabs = ["Num of pairs: %s" % min(nybinned),"Num of pairs: %s"% max(nybinned)]
        for j in range(len(xbinned)):
            if j == np.argmin(mysize) or j == np.argmax(mysize):
                axs[i].plot(xbinned[j], ybinned[j],linestyle="None",alpha=0.5,marker="o", markersize=mysize[j],color="red",label ="Num of pairs: %s"%  nybinned[j])
            else:
                axs[i].plot(xbinned[j], ybinned[j],linestyle="None",alpha=0.5,marker="o", markersize=mysize[j],color="red")
        axs[i].errorbar(xbinned, ybinned,eybinned, capsize=5)
        axs[i].set(ylabel=labsyay[i])
        axs[i].grid()
        # Computing the linear fit to the data, using the amount of
        x = xbinned.reshape((-1, 1))
        y = ybinned
        model = LinearRegression().fit(x, y,nybinned)
        r_sq = model.score(x, y)
        slope = model.coef_
        y_pred = model.intercept_ + model.coef_ * x.ravel()
        axs[i].plot(xbinned, y_pred,'-')
        #sn.regplot(x=xaxall, y=yayall, ax=axs[i])
        if i==0:
            xmin = np.amin(xbinned)
            xmax = np.amax(xbinned)
            xprang = (xmax - xmin) * 0.03
            x0 = xmin+xprang
#        y0, yf = axs[i].get_ylim()
#        my0 = y0-(yf-y0)*0.13
        #new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x0,xf = xlim[k]
        y0,yf = ylim[i]
        xprang = xf-x0
        yprang = yf-y0
        my0 = y0-(yf-y0)*0.13
        axs[i].set(xlim=(x0,xf))
        axs[i].set(ylim=(y0,yf))
        axs[i].text(x0, y0+yprang*0.05, 'R^2: %6.2f' % (r_sq),fontsize=8)
        axs[i].text(xf-xprang*0.3,y0+yprang*0.05,'Slope %5.2f' % slope,fontsize=8)
#        axs[i].set(ylim=(y0-(yf-y0)*0.15,yf+(yf-y0)*0.15))
#        axs[i].set(xlim=(xmin - xprang*3, xmax + xprang*3))
    axs[0].legend(prop={'size': 9})
    axs[6].set(xlabel=labsxax[k])
    axs[7].set(xlabel=labsxax[k])
    pdf4.savefig(fig)
    plt.close()

pdf4.close()



#=====================================================================
#Plotting LHa vs Mco for each galaxy, highlighting the top ten.

print "Plotting Smol-Mco for each galaxy"

mycol=['grey','purple','blue','green','orange','red', 'pink', 'orange']
spring = cm.get_cmap('spring',256)
mspring   = ListedColormap(spring(np.linspace(0.1, 0.8, 256)))
mycmp=['Greys','Purples','Blues','Greens','Oranges','Reds',mspring,'Wistia']

xaxarray = [Sigmamoleover,aviriaGMCover]
yayarray = [SizepcGMCover,sigmavGMCover,MasscoGMCover]
labsxax = [r'log($\Sigma_{\rm mol}$) [M$_{\odot}$ pc$^{-2}$]',r'log($\alpha_{\rm vir}$]']
labsyay = ['log(GMC size) [pc]', r'log($\sigma_{\rm v}$ GMC) [km/s]', r'log(Ma$_{\rm CO}$) [M$_{\odot}$]']

pdf = fpdf.PdfPages("Smol_Mco.pdf")
marker_style = dict(markersize=4)
for i in range(len(xaxarray)):
    for k in range(len(yayarray)):
        fig, axs = plt.subplots(4, 2,sharex='col',figsize=(8,10),gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace = 0.3)
        r, g, b = np.random.uniform(0, 1, 3)
        fig.suptitle(r'All galaxies - $\Sigma_{mol}$  vs CO masses', fontsize=14,va='top')
        axs = axs.ravel()
        for j in range(len(galaxias)):
            xaxt = np.log10(xaxarray[i][j])
            yayt = np.log10(yayarray[k][j])
            idok = np.where((yayt == yayt) &  (xaxt == xaxt))
            xax = xaxt[idok] ; yay = yayt[idok]
            xax,yay=checknaninf(xax,yay,100000,100000)
            axs[j].plot(xax, yay, '8', alpha=0.4, color=mycol[j],label='%s' % galaxias[j],**marker_style)
            if len(xax) > 3:
               sns.kdeplot(xax,yay, ax=axs[j],n_levels=3,shade=True,shade_lowest=False,alpha=0.4,cmap=mycmp[j])
               sns.kdeplot(xax,yay, ax=axs[j],linewidths=0.5,cmap=mycmp[j],n_levels=3,shade_lowest=False)
            axs[j].grid()
            axs[j].legend(prop={'size': 5})
            axs[j].set(ylabel=labsyay[k])
        axs[6].set(xlabel=labsxax[i])
        axs[7].set(xlabel=labsxax[i])
        pdf.savefig(fig)
        plt.close()

pdf.close()
#exit()

#---------------------------------------------------
#Plotting HII regions prop vs num of GMCs associated.
print "Plotting HII regions prop vs num of GMCs associated."
marker_style  = dict(markersize=2)
# numGMConHII[i][j][k]
# i: number of galaxies
# j = 0 -> number of regions < size HII * 2 ;  j = 1 -> array with indices of the GMCs associated.
# k: indices of HII regions

# LumHacorr
GMCprop = [DisHIIGMCover,SizepcGMCover,sigmavGMCover,MasscoGMCover,aviriaGMCover,Sigmamoleover,TpeakGMCover,tauffGMCover]
shortlab = ['HIIGMCdist','GMCsize', 'sigmav', 'Mco','avir','Smol','TpeakCO','tauff']
arrayxax = [GaldisHII,SizepcHII,LumHacorr,sigmavHII,metaliHII,varmetHII]
arrayyay =  numGMConHII
#[DisHIIGMCover,SizepcGMCover,sigmavGMCover,MasscoGMCover,aviriaGMCover,Sigmamoleover,TpeakGMCover,tauffGMCover]
labsxax = ['Galactocentric radius [kpc]','log(HII region size) [pc]', r'log(Luminosity H$\alpha$) [erg/s]',r'log($\sigma_{v}$ HII region) [km/s]','Metallicity','Metallicity variation']
labsyay = "Number of GMCs < size HII R"
#labsyay = ['log(Distance  HII-GMC) [pc]','log(GMC size) [pc]',r'log($\sigma_v$) [km/s]',r'log(Mass$_{CO}$) [10^5 M$_{\odot}$]',r'log($\alpha_{vir}$)',r'log($\Sigma_{mol}$)',r'log(CO $T_{peak}$ [K])',r'log($\tau_{ff}$) [yr]']

minmass = np.min([np.concatenate([f.tolist() for f in MasscoGMC])][0])
maxmass = np.max([np.concatenate([f.tolist() for f in MasscoGMC])][0])


pdf3 = fpdf.PdfPages("Num_GMCs_galbygal%s.pdf" % namegmc)  # type: PdfPages
for k in range(len(arrayxax)):
    print 'Plotting %s' % labsxax[k]
    sns.set(style='white',color_codes=True)
    # Galactic distance vs: Mco, avir, sigmav,Sigmamol
    fig, axs = plt.subplots(4, 2,sharex='col',figsize=(8,10),gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace = 0.3)
    
    fig.suptitle('All galaxies - Overlapping HIIregions and GMCs\nColor coded %s' % shortlab[k], fontsize=15,va='top') #labprop
    axs = axs.ravel()
    for j in range(len(galaxias)):
        xax = [h for h in arrayxax[k][j]]
        yay = [h for h in arrayyay[j][0]]
        idon = np.where(arrayyay[j][0] > 1)
        xaxon = np.array(xax)[idon] ; yayon = np.array(yay)[idon]
        myay = [MasscoGMC[j][f].tolist() for f in arrayyay[j][1] if len(f) > 1]
        rmsyay = []
        meanyay = []
        for myv in myay:
            rmsyay.append(np.std(myv))
            meanyay.append(np.mean(myv))
        if len(rmsyay)>0:
            rmsyay  = rmsyay/np.amax(rmsyay)
        if k < 4:
            xax = np.log10(xax)
            xaxon = np.log10(xaxon)
        axs[j].plot(xax, yay,'.',label='%s'%galaxias[j],alpha=0.2,**marker_style)
        myp = axs[j].scatter(xaxon, yayon,marker='.',s=60,c=np.log10(rmsyay),alpha=1,cmap='binary',edgecolor='black',linewidths=0.2)
        #axs[j].errorbar(xaxon, yayon, rmsyay, marker='.',capsize=5,**marker_style)
        fig.colorbar(myp, ax=axs[j],shrink=0.9)
        axs[j].grid()
        axs[j].legend(prop={'size': 5})
    axs[0].set(ylabel=labsyay)
    axs[6].set(xlabel=labsxax[k])
    axs[7].set(xlabel=labsxax[k])
    pdf3.savefig(fig)
    plt.close()

pdf3.close()

print "Total number of GMCs for which more than 1 are associated to the same HII region"

for i in range(8):
    myids = np.where(numGMConHII[i][0]>1)[0]
    print galaxias[i],np.sum(numGMConHII[i][0][myids])

#=========================
#Heyerr relationship
rootR = [np.sqrt(f) for f in SizepcGMCover]
arrayyay=  np.divide(sigmavGMCover,rootR)
arrayxax = Sigmamoleover #
#arrayxax=np.multiply(Sigmamoleover,aviriaGMCover)

labsyay = r'log($\sigma_v$/R$^{0.5}$) [km/s pc$^{-1/2}$]'
labsxax = r'log($\Sigma_{mol}$[M$_{\odot}$/pc$^2$])'#

pdf6 = fpdf.PdfPages("Correlations_Heyer_allgals_GMC_%s.pdf" % namegmc)  # type: PdfPages

#print "Starting loop to create figures of all galaxies together - vs avir val"

sns.set(style='white',color_codes=True)
fig, axs = plt.subplots(1, 1,sharex='col',figsize=(9,10),dpi=80,gridspec_kw={'hspace': 0})
plt.subplots_adjust(wspace = 0.3)
fig.suptitle('All galaxies - Overlapping HIIregions and GMCs', fontsize=18,va='top')
yaytmp = arrayyay ; xaxtmp = arrayxax
xaxall = np.concatenate([f.tolist() for f in xaxtmp])
yayall = np.concatenate([f.tolist() for f in yaytmp])
xaxall = np.log10(xaxall)
yayall = np.log10(yayall)
idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
xaxall = xaxall[idok] ; yayall = yayall[idok]
lim1 = np.nanmedian(xaxall)-np.nanstd(xaxall)*4
lim2 = np.nanmedian(xaxall)+np.nanstd(xaxall)*4
indlim = np.where((xaxall < lim2) & (xaxall>lim1))
xaxall = xaxall[indlim] ; yayall = yayall[indlim]
    
for j in range(len(galaxias)):
       xax2 = [h for h in arrayxax[j]]
       yay2 = [h for h in arrayyay[j]]
       xax = np.log10(xax2)
       yay = np.log10(yay2)
       axs.plot(xax, yay,'8',label='%s'%galaxias[j],alpha=0.7,markersize=5)

       
axs.set(ylabel=labsyay)
#axs.set_yscale('log')
#axs.set_xscale('log')
axs.grid()

ybc = np.log10(math.sqrt(math.pi*ct.G.cgs.value/5*ct.M_sun.cgs.value/ct.pc.cgs.value*10**-10))+0.5*xaxall
axs.plot(xaxall, ybc)

xmin = np.amin(xaxall)
xmax = np.amax(xaxall)
xprang = (xmax - xmin) * 0.1
x = xaxall.reshape((-1, 1))
y = yayall
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
y_pred = model.intercept_ + model.coef_ * x.ravel()

axs.plot(xaxall, y_pred,'-')
        #sn.regplot(x=xaxall, y=yayall, ax=axs[i])
x0 = xmin+xprang
x0, xf = axs.get_xlim()
y0, yf = axs.get_ylim()
#x0,xf = xlim[k]
#y0,yf = ylim[i]
axs.text(x0, y0, 'R sq: %6.4f' % (r_sq))
#        axs[i].set(xlim=(xmin - xprang, xmax + xprang))
axs.set(xlim=(x0,xf))
axs.set(ylim=(y0,yf))
axs.legend(prop={'size': 14})
axs.set(xlabel=labsxax)
pdf6.savefig(fig)
plt.close()

pdf6.close()

#    mybin = 20
#    xbinned,ybinned,eybinned,nybinned=bindata(xaxall,yayall,mybin)
#    # if there is any nan inside
#    ido = np.where(np.array(nybinned) != 0)
#    xbinned  = xbinned[ido]
#    ybinned  = [g for g in np.array(ybinned)[ido]]
#    eybinned = [g for g in np.array(eybinned)[ido]]
#    nybinned = [g for g in np.array(nybinned)[ido]]
#    # Plot binned data
#    mysize = np.array(nybinned).astype(float)
#    mysize = (mysize-np.min(mysize))/(np.max(mysize)-np.min(mysize))*9+3
#    mylims = [np.argmin(mysize),np.argmax(mysize)]
#    mylabs = ["Num of pairs: %s" % min(nybinned),"Num of pairs: %s"% max(nybinned)]
#    #pdb.set_trace()
#    for j in range(len(xbinned)):
#        if j == np.argmin(mysize) or j == np.argmax(mysize):
#            axs[i].plot(xbinned[j], ybinned[j],linestyle="None",alpha=0.5,marker="o", markersize=mysize[j],color="red",label ="Num of pairs: %s"%  nybinned[j])
#        else:
#            axs[i].plot(xbinned[j], ybinned[j],linestyle="None",alpha=0.5,marker="o", markersize=mysize[j],color="red")
#    axs[i].errorbar(xbinned, ybinned,eybinned, capsize=5)
#    axs[i].set(ylabel=labsyay[i])
#    axs[i].grid()
#    # Computing the linear fit to the data, using the amount of
#    xmin = np.amin(xbinned)
##    xmax = np.amax(xbinned)
#    xprang = (xmax - xmin) * 0.03
#    x = xbinned.reshape((-1, 1))
#    y = ybinned
#    model = LinearRegression().fit(x, y,nybinned)
#    r_sq = model.score(x, y)
#    y_pred = model.intercept_ + model.coef_ * x.ravel()
#    axs[i].plot(xbinned, y_pred,'-')
#    #sn.regplot(x=xaxall, y=yayall, ax=axs[i])
#    x0 = xmin+xprang
#    y0, yf = axs[i].get_ylim()
#    my0 = y0-(yf-y0)*0.13
#3    axs[i].text(x0, my0, 'R^2: %6.4f' % (r_sq),fontsize=10)
##    axs[i].set(ylim=(y0-(yf-y0)*0.15,yf+(yf-y0)*0.15))
##    axs[i].set(xlim=(xmin - xprang*3, xmax + xprang*3))
##
#axs[0].legend(prop={'size': 9})
#axs[4].set(xlabel=labsxax)
#pdf5.savefig(fig)
#plt.close()
#
#pdf5.close()

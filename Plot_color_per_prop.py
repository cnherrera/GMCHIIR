import seaborn as sn
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf


#=====================================================
def outinf(v1,v2):
    v1n = np.array(v1)
    v2n = np.array(v2)
    v1n2 = v1n[np.isfinite(v1n)]
    v2n2 = v2n[np.isfinite(v1n)]
    v1n3 = v1n2[np.isfinite(v2n2)]
    v2n3 = v2n2[np.isfinite(v2n2)]

    return v1n3,v2n3

#==========================================================================================================
dirhii = "../../Catalogs-HIIregions/hii_region_enhanced_catalogs/"
dirgmc = "../../Catalogs-CPROPS/"
diralma = '../../ALMA-LP-delivery/delivery_v3p3/'

# The kind of data we will plot, native or homogenized data.
namegmc = "_12m+7m+tp_co21_150pc_props" # native

#--------------------------------------------------------------------------------
galaxias,GMCprop,HIIprop,RAgmc,DECgmc,RAhii,DEChii,labsxax,labsyay= pickle.load(open(('Galaxies_variables_GMC%s.pickle' % namegmc),"rb"))
GaldisHIIover,SizepcHIIover,LumHacorrover,sigmavHIIover,ratlin,metaliHIIover,varmetHIIover = HIIprop
DisHIIGMCover,MasscoGMCover,SizepcGMCover,Sigmamoleover,sigmavGMCover,aviriaGMCover,TpeakGMCover,tauffGMCover = GMCprop
shortlab = ['HIIGMCdist', 'Mco','GMCsize','Smol', 'sigmav','avir','TpeakCO','tauff']
MassesCO = [1e5*i for i in MasscoGMCover] #

# Limits in the properties of HIIR and GMCs
xlim,ylim,xx,yy=pickle.load(open('limits_properties.pickle',"rb"))

#=================================================

# Adding colors wrt to GMC properties
for i,lab in enumerate(labsyay):
    pdf1 = fpdf.PdfPages("Correlations_allgals_color-%s_GMC%s.pdf" % (shortlab[i],namegmc))  # type: PdfPages
    for k,xax0 in enumerate(HIIprop):
        sn.set(style='white',color_codes=True)
        fig, axs = plt.subplots(4, 2,sharex='col',figsize=(8,10),gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace = 0.3)
        fig.suptitle('Correlation overlapping HIIregions and GMCs\nColor coded %s' % lab, fontsize=15,va='top')
        axs = axs.ravel()
        for j,yay0 in enumerate(GMCprop):
            xaxall = np.concatenate([f.tolist() for f in xax0])
            yayall = np.concatenate([f.tolist() for f in yay0])
            myvar = np.concatenate([f.tolist() for f in GMCprop[i]])        #---------------------------------------------
            if k<5:
                xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)
            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok] ; yayall = yayall[idok]
            myvar = myvar[idok]                                            #---------------------------------------
            lim1=np.nanmedian(xaxall)-np.nanstd(xaxall)*4
            lim2=np.nanmedian(xaxall)+np.nanstd(xaxall)*4
            indlim = np.where((xaxall < lim2) & (xaxall>lim1))
            xaxall = xaxall[indlim] ; yayall = yayall[indlim]
            myvar = myvar[indlim]
            myy = []
            myx = []
            for l in range(len(galaxias)):
                yay = yay0[l]
                xax = xax0[l]
                if k < 5:
                    xax = np.log10(xax)
                yay = np.log10(yay)
                myy.append(yay)
                myx.append(xax)
            myx = np.concatenate([f.tolist() for f in myx])
            myy = np.concatenate([f.tolist() for f in myy])
            myy,myx = outinf(myy,myx)
            axs[j].set(ylabel=labsyay[j])
            axs[j].grid()
            x = myx.reshape((-1, 1))
            y = myy
            model = LinearRegression().fit(x, y)
            r_sq = model.score(x, y)
            slope = model.coef_
            y_pred = model.intercept_ + model.coef_ * x.ravel()
            axs[j].scatter(xaxall, yayall,marker='.',s=10,c=np.log10(myvar),cmap='gnuplot2_r')#viridis')
            axs[j].plot(myx, y_pred, '-')
            x0,xf = xlim[k]
            y0,yf = ylim[j]
            xprang = (xf - x0) * 0.1
            yprang = (yf - y0) * 0.05
            axs[j].text(xf-3*xprang , y0+yprang, 'Slope %5.2f' % slope, fontsize=10)
            axs[j].text(x0+xprang/3,y0+yprang, 'R sq: %5.2f' % (r_sq), fontsize=10)
            axs[j].set(xlim=(x0,xf))
            axs[j].set(ylim=(y0,yf))
        axs[0].legend(prop={'size': 5})
        axs[6].set(xlabel=labsxax[k])
        axs[7].set(xlabel=labsxax[k])
        pdf1.savefig(fig)
        plt.close()
    pdf1.close()




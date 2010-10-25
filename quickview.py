import idlsave
import numpy as np
import matplotlib.pyplot as plt
import savunpack
import matplotlib.mlab as mlab


junk = idlsave.read('Stars/keck-fit-lite.sav')
stars = junk['stars']

o_staterr = savunpack.savunpack(stars.o_staterr)
c_staterr = savunpack.savunpack(stars.c_staterr)
smeabund  = savunpack.savunpack(stars.smeabund)
goodidx = np.where( (stars.o_abund > 0) & (stars.c_abund > 0) &
                        (o_staterr[:,0] > -0.3) &(c_staterr[:,0] > -0.3))

stars.o_abund = stars.o_abund - 8.7
stars.c_abund = stars.c_abund - 8.5
stars = stars[goodidx]
o_staterr = o_staterr[goodidx]
c_staterr = c_staterr[goodidx]
smeabund = smeabund[goodidx,:]
smeabund = smeabund.reshape(619,100)
#oulstars = stars[np.where((o_staterr[:,0] < -0.3) & (c_staterr[:,0] > -0.3))]
#culstars = stars[np.where((c_staterr[:,0] < -0.3) & (o_staterr[:,0] > -0.3))]

#put errorbars into a form that staterr likes

o_staterr = np.abs(o_staterr.transpose())
c_staterr = np.abs(c_staterr.transpose())
ni_abund  = smeabund[:,27]  + np.log10(smeabund[:,0])

#plt.plot(stars.o_abund-8.7,stars.c_abund-8.5,'o')
#plt.errorbar(stars.o_abund,stars.c_abund,xerr=o_staterr,yerr=c_staterr,marker='o',ls='None',color='black')

plt.clf()
#plt.scatter(stars.o_abund-8.7,stars.c_abund-8.5,marker='o',color='black')
#x = np.linspace(-1,1,10)
#plt.plot(x,x)
#plt.plot(x,x+0.2)

#r = np.core.rec.fromarrays(
#    [stars.o_abund,stars.c_abund,stars.teff,stars.feh,
#     stars.vsini,stars.logg,ni_abund],
#    names=[
#        'o','c','teff','feh',
#        'vsini','logg','ni'])

#mlab.rec2csv(r,'test.csv',delimiter=',')



#type c_staterr


plt.show()

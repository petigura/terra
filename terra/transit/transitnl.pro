pro transitnl,c1,c2,c3,c4,b1=b1,teff=teff,lg1=lg1,mh1=mh1
close,1
; This routine takes the limb-darkening coefficients of Claret
; and computes lightcurves for any set of spectral parameters.
; The atlas code has the following temperatures:
T=[ 3500., 3750., 4000., 4250., 4500., 4750., 5000., 5250., 5500., 5750., 6000., 6250.,$
    6500., 6750., 7000., 7250., 7500., 7750., 8000., 8250., 8500., 8750., 9000., 9250.,$
    9500., 9750.,10000.,10250.,10500.,10750.,11000.,11250.,11500.,11750.,12000.,12250.,$
   12500.,12750.,13000.,14000.,15000.,16000.,17000.,19000.,20000.,21000.,22000.,23000.,$
   24000.,25000.,26000.,27000.,28000.,29000.,30000.,31000.,32000.,33000.,34000.,35000.,$
   36000.,37000.,38000.,39000.,40000.,41000.,42000.,43000.,44000.,45000.,46000.,47000.,$
   48000.,49000.,50000.]
nt=n_elements(T)
; The atlas code has the following log(g):
lgg=[0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]
ng=n_elements(lgg)
; The atlas code has the following [M/H]:
mh=[-5.,-4.5,-4.,-3.5,-3.,-2.5,-2.,-1.5,-1.,-0.5,-0.3,-0.2,-0.1,0.,0.1,0.2,0.3,0.5,1.]
nm=n_elements(mh)
; Total number of atmospheres:
;print,nm,ng,nt,nm*ng*nt
; Photometric bands:
band=['u','v','b','y','U','B','V','R','I','J','H','K']
if(n_elements(b1) eq 0) then read,'Band? 0=u,1=v,2=b,3=y,4=U,5=B,6=V,7=R,8=I,9=J,10=H,11=K ',b1
if(n_elements(teff) eq 0) then read,'Teff? 3500-50000 K ',teff
if(n_elements(lg1) eq 0) then read,'log(g)? 0-5 (cm/s^2) ',lg1
if(n_elements(mh1) eq 0) then read,'metallicity? [M/H] -5 to 1 ',mh1
mtmp=min(abs(t-teff),itmp)
mmt=min(abs(mh-mh1),imh)
mlgg=min(abs(lg1-lgg),ilgg)
print,'Using Teff=',t(itmp),', [M/H]=',mh(imh),', log(g)=',lgg(ilgg)
t0=t(itmp)
m0=mh(imh)
g0=lgg(ilgg)
data1=dblarr(17)
data2=dblarr(17)
data3=dblarr(17)
data4=dblarr(17)
openr,1,'atlas.dat'
while (not eof(1)) do begin
  readf,1,data1
  readf,1,data2
  readf,1,data3
  readf,1,data4
;   print,data1(2),data1(3),data1(4)
  if(data1(2) eq g0 and data1(3) eq t0 and data1(4) eq m0) then begin
    c1=data1(4+b1)
    c2=data2(4+b1)
    c3=data3(4+b1)
    c4=data4(4+b1)
    print,'Coefficients are ',c1,c2,c3,c4
    goto,finished
  endif
endwhile
print,'No atmospheres with those parameters were found'
c1=0.d0
c2=0.d0
c3=0.d0
c4=0.d0
finished:
close,1
return
end

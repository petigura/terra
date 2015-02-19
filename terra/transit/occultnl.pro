pro occultnl,rl,c1,c2,c3,c4,b0,mulimb0,mulimbf,plotquery,_extra=e
; Please cite Mandel & Agol (2002) if making use of this routine.
timing=systime(1)
; This routine uses the results for a uniform source to
; compute the lightcurve for a limb-darkened source
; (5-1-02 notes)
;Input:
;  rl        radius of the lens   in units of the source radius
;  c1-c4     limb-darkening coefficients
;  b0        impact parameter normalized to source radius
;  plotquery =1 to plot magnification,  =0 for no plots
;  _extra=e  plot parameters
;Output:
; mulimb0 limb-darkened magnification
; mulimbf lightcurves for each component
; 
; First, make grid in radius:
; Call magnification of uniform source:
occultuniform,b0,rl,mulimb0
bt0=b0
fac=max(abs(mulimb0-1.d0))
;print,rl
omega=4.d0*((1.d0-c1-c2-c3-c4)/4.d0+c1/5.d0+c2/6.d0+c3/7.d0+c4/8.d0)
nb=n_elements(b0)
indx=where(mulimb0 ne 1.d0)
mulimb=mulimb0(indx)
mulimbf=dblarr(nb,5)
mulimbf(*,0)=mulimbf(*,0)+1.d0
mulimbf(*,1)=mulimbf(*,1)+0.8d0
mulimbf(*,2)=mulimbf(*,2)+2.d0/3.d0
mulimbf(*,3)=mulimbf(*,3)+4.d0/7.d0
mulimbf(*,4)=mulimbf(*,4)+0.5d0
nr=2
dmumax=1.d0
;while (dmumax gt fac*1.d-10 and nr le 16) do begin
while (dmumax gt fac*1.d-3) do begin
;while (dmumax gt 1.d-6 and nr le 4) do begin
  mulimbp=mulimb
  nr=nr*2
  dt=0.5d0*!pi/double(nr)
  t=dt*dindgen(nr+1)
  th=t+0.5d0*dt
  r=sin(t)
  sig=sqrt(cos(th(nr-1)))
  mulimbhalf =sig^3*mulimb0(indx)/(1.d0-r(nr-1))
  mulimb1    =sig^4*mulimb0(indx)/(1.d0-r(nr-1))
  mulimb3half=sig^5*mulimb0(indx)/(1.d0-r(nr-1))
  mulimb2    =sig^6*mulimb0(indx)/(1.d0-r(nr-1))
  for i=1,nr-1 do begin
; Calculate uniform magnification at intermediate radii:
    occultuniform,b0(indx)/r(i),rl/r(i),mu
; Equation (29):
    sig1=sqrt(cos(th(i-1)))
    sig2=sqrt(cos(th(i)))
    mulimbhalf =mulimbhalf +r(i)^2*mu*(sig1^3/(r(i)-r(i-1))-sig2^3/(r(i+1)-r(i)))
    mulimb1    =mulimb1    +r(i)^2*mu*(sig1^4/(r(i)-r(i-1))-sig2^4/(r(i+1)-r(i)))
    mulimb3half=mulimb3half+r(i)^2*mu*(sig1^5/(r(i)-r(i-1))-sig2^5/(r(i+1)-r(i)))
    mulimb2    =mulimb2    +r(i)^2*mu*(sig1^6/(r(i)-r(i-1))-sig2^6/(r(i+1)-r(i)))
  endfor
  mulimb=((1.d0-c1-c2-c3-c4)*mulimb0(indx)+c1*mulimbhalf*dt+c2*mulimb1*dt+$
           c3*mulimb3half*dt+c4*mulimb2*dt)/omega
  ix1=where(mulimb+mulimbp ne 0.d0)
  dmumax=max(abs(mulimb(ix1)-mulimbp(ix1))/(mulimb(ix1)+mulimbp(ix1)))
;  print,'Difference ',dmumax,' nr ',nr
endwhile
mulimbf(indx,0)=mulimb0(indx)
mulimbf(indx,1)=mulimbhalf*dt
mulimbf(indx,2)=mulimb1*dt
mulimbf(indx,3)=mulimb3half*dt
mulimbf(indx,4)=mulimb2*dt
mulimb0(indx)=mulimb
if(plotquery eq 1) then plot,bt0,mulimb0,_extra=e
if(plotquery eq 1) then oplot,bt0,mulimbf(*,0),linestyle=2
b0=bt0
print,'Time ',systime(1)-timing
return
end

pro occultuniform,b0,w,muo1
if(abs(w-0.5d0) lt 1.d-3) then w=0.5d0
; This routine computes the lightcurve for occultation
; of a uniform source without microlensing  (Mandel & Agol 2002).
;Input:
;
; rs   radius of the source (set to unity)
; b0   impact parameter in units of rs
; w    occulting star size in units of rs
;
;Output:
; muo1 fraction of flux at each b0 for a uniform source
;
; Now, compute pure occultation curve:
nb=n_elements(b0)
muo1=dblarr(nb)
for i=0,nb-1 do begin
; substitute z=b0(i) to shorten expressions
z=b0(i)
; the source is unocculted:
; Table 3, I.
if(z ge 1.d0+w) then begin
  muo1(i)=1.d0
  goto,next
endif
; the  source is completely occulted:
; Table 3, II.
if(w ge 1.d0 and z le w-1.d0) then begin
  muo1(i)=0.d0
  goto,next
endif
; the source is partly occulted and the occulting object crosses the limb:
; Equation (26):
if(z ge abs(1.d0-w) and z le 1.d0+w) then begin
  kap1=acos(min([(1.d0-w^2+z^2)/2.d0/z,1.d0]))
  kap0=acos(min([(w^2+z^2-1.d0)/2.d0/w/z,1.d0]))
  lambdae=w^2*kap0+kap1
  lambdae=(lambdae-0.5d0*sqrt(max([4.d0*z^2-(1.d0+z^2-w^2)^2,0.d0])))/!pi
  muo1(i)=1.d0-lambdae
endif
; the occulting object transits the source star (but doesn't
; completely cover it):
if(z le 1.d0-w) then muo1(i)=1.d0-w^2
next:
endfor
;muo1=1.d0-lambdae
return
end

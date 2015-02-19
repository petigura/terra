pro occultquad,z0,u1,u2,p,muo1,mu0
if(abs(p-0.5d0) lt 1.d-3) then p=0.5d0
;  This routine computes the lightcurve for occultation
;  of a quadratically limb-darkened source without microlensing.
;  Please cite Mandel & Agol (2002) if you make use of this routine
;  in your research.  Please report errors or bugs to agol@tapir.caltech.edu
;
; Input:
;
; rs   radius of the source (set to unity)
; z0   impact parameter in units of rs
; p    occulting star size in units of rs
; u1   linear    limb-darkening coefficient (gamma_1 in paper)
; u2   quadratic limb-darkening coefficient (gamma_2 in paper)
;
; Output:
;
; muo1 fraction of flux at each z0 for a limb-darkened source
; mu0  fraction of flux at each z0 for a uniform source
;
; Limb darkening has the form:
;  I(r)=[1-u1*(1-sqrt(1-(r/rs)^2))-u2*(1-sqrt(1-(r/rs)^2))^2]/(1-u1/3-u2/6)/pi
; 
; To use this routine
;
; Now, compute pure occultation curve:
nz=n_elements(z0)
mu=dblarr(nz)
lambdad=dblarr(nz)
etad=dblarr(nz)
lambdae=dblarr(nz)
muo1=dblarr(nz)
; Loop over each impact parameter:
for i=0,nz-1 do begin
; substitute z=z0(i) to shorten expressions
z=z0(i)
x1=(p-z)^2
x2=(p+z)^2
x3=p^2-z^2
; the source is unocculted:
; Table 3, I.
if(z ge 1.d0+p) then begin
  lambdad(i)=0.d0
  etad(i)=0.d0
  lambdae(i)=0.d0
  goto,next
endif
; the  source is completely occulted:
; Table 3, II.
if(p ge 1.d0 and z le p-1.d0) then begin
  lambdad(i)=1.d0
  etad(i)=1.d0
  lambdae(i)=1.d0
  goto,next
endif
; the source is partly occulted and the occulting object crosses the limb:
; Equation (26):
if(z ge abs(1.d0-p) and z le 1.d0+p) then begin
  kap1=acos(min([(1.d0-p^2+z^2)/2.d0/z,1.d0]))
  kap0=acos(min([(p^2+z^2-1.d0)/2.d0/p/z,1.d0]))
  lambdae(i)=p^2*kap0+kap1
  lambdae(i)=(lambdae(i)-0.5d0*sqrt(max([4.d0*z^2-(1.d0+z^2-p^2)^2,0.d0])))/!pi
  if(z eq 1.d0-p) then begin
  endif
endif
; the occulting object transits the source star (but doesn't
; completely cover it):
if(z le 1.d0-p) then lambdae(i)=p^2
; the edge of the occulting star lies at the origin- special 
; expressions in this case:
if(abs(z-p) lt 1.d-4*(z+p)) then begin
; Table 3, Case V.:
  if(z ge 0.5d0) then begin
    lam=0.5d0*!pi
    q=0.5d0/p
    Kk=ellk(q)
    Ek=ellec(q)
; Equation 34: lambda_3
    lambdad(i)=1.d0/3.d0+16.d0*p/9.d0/!pi*(2.d0*p^2-1.d0)*Ek-$
               (32.d0*p^4-20.d0*p^2+3.d0)/9.d0/!pi/p*Kk
; Equation 34: eta_1
    etad(i)=1.d0/2.d0/!pi*(kap1+p^2*(p^2+2.d0*z^2)*kap0-$
            (1.d0+5.d0*p^2+z^2)/4.d0*sqrt((1.d0-x1)*(x2-1.d0)))
    if(p eq 0.5d0) then begin
; Case VIII: p=1/2, z=1/2
      lambdad(i)=1.d0/3.d0-4.d0/!pi/9.d0
      etad(i)=3.d0/32.d0
    endif
    goto,next
  endif else begin
; Table 3, Case VI.:
    lam=0.5d0*!pi
    q=2.d0*p
    Kk=ellk(q)
    Ek=ellec(q)
; Equation 34: lambda_4
    lambdad(i)=1.d0/3.d0+2.d0/9.d0/!pi*(4.d0*(2.d0*p^2-1.d0)*Ek+$
               (1.d0-4.d0*p^2)*Kk)
; Equation 34: eta_2
    etad(i)=p^2/2.d0*(p^2+2.d0*z^2)
    goto,next
  endelse
endif
; the occulting star partly occults the source and crosses the limb:
; Table 3, Case III:
if((z gt 0.5d0+abs(p-0.5d0) and z lt 1.d0+p) or (p gt 0.5d0 and z gt $
    abs(1.d0-p)*1.0001d0 and z lt p)) then begin
  lam=0.5d0*!pi
  q=sqrt((1.d0-(p-z)^2)/4.d0/z/p)
  Kk=ellk(q)
  Ek=ellec(q)
  n=1.d0/x1-1.d0
    Pk=Kk-n/3.d0*rj(0.d0,1.d0-q^2,1.d0,1.d0+n)
; Equation 34, lambda_1:
  lambdad(i)=1.d0/9.d0/!pi/sqrt(p*z)*(((1.d0-x2)*(2.d0*x2+x1-3.d0)-$
      3.d0*x3*(x2-2.d0))*Kk+4.d0*p*z*(z^2+7.d0*p^2-4.d0)*Ek-3.d0*x3/x1*Pk)
  if(z lt p) then lambdad(i)=lambdad(i)+2.d0/3.d0
; Equation 34, eta_1:
  etad(i)=1.d0/2.d0/!pi*(kap1+p^2*(p^2+2.d0*z^2)*kap0-$
        (1.d0+5.d0*p^2+z^2)/4.d0*sqrt((1.d0-x1)*(x2-1.d0)))
  goto,next
endif
; the occulting star transits the source:
; Table 3, Case IV.:
if(p le 1.d0 and z le (1.d0-p)*1.0001d0) then begin
  lam=0.5d0*!pi
  q=sqrt((x2-x1)/(1.d0-x1))
  Kk=ellk(q)
  Ek=ellec(q)
  n=x2/x1-1.d0
    Pk=Kk-n/3.d0*rj(0.d0,1.d0-q^2,1.d0,1.d0+n)
; Equation 34, lambda_2:
  lambdad(i)=2.d0/9.d0/!pi/sqrt(1.d0-x1)*((1.d0-5.d0*z^2+p^2+x3^2)*Kk+$
             (1.d0-x1)*(z^2+7.d0*p^2-4.d0)*Ek-3.d0*x3/x1*Pk)
  if(z lt p) then lambdad(i)=lambdad(i)+2.d0/3.d0
  if(abs(p+z-1.d0) le 1.d-4) then begin
    lambdad(i)=2/3.d0/!pi*acos(1.d0-2.d0*p)-4.d0/9.d0/!pi*$
          sqrt(p*(1.d0-p))*(3.d0+2.d0*p-8.d0*p^2)
  endif
; Equation 34, eta_2:
  etad(i)=p^2/2.d0*(p^2+2.d0*z^2)
endif
next:
endfor
; Now, using equation (33):
omega=1.d0-u1/3.d0-u2/6.d0
muo1=1.d0-((1.d0-u1-2.d0*u2)*lambdae+(u1+2.d0*u2)*lambdad+u2*etad)/omega
; Equation 25:
mu0=1.d0-lambdae
return
end

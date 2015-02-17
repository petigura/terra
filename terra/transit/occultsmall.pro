pro occultsmall,p,c1,c2,c3,c4,z,mu
; This routine approximates the lightcurve for a small 
; planet. (See section 5 of Mandel & Agol (2002) for
; details).  Please cite Mandel & Agol (2002) if making
; use of this routine.
; Input:
;  p      ratio of planet radius to stellar radius
;  c1-c4  non-linear limb-darkening coefficients
;  z      impact parameters (normalized to stellar radius)
;        - this is an array which must be input to the routine
; Output:
;  mu     flux relative to unobscured source for each z
;
nb=n_elements(z)
mu=dblarr(nb)+1.d0
indx=where(z gt 1.-p and z lt 1.+p)
i1=1.-c1-c2-c3-c4
norm=!pi*(1.-c1/5.-c2/3.-3.*c3/7.-c4/2.)
sig=sqrt(sqrt(1.-(1.-p)^2))
x=1.-(z(indx)-p)^2
tmp=(1.-c1*(1.-4./5.*x^0.25)-c2*(1.-2./3.*x^0.5)$
  -c3*(1.-4./7.*x^0.75)-c4*(1.-4./8.*x))
mu(indx)=1.-tmp*(p^2*acos((z(indx)-1.)/p)$
  -(z(indx)-1.)*sqrt(p^2-(z(indx)-1.)^2))/norm
indx=where(z le 1.-p and z ne 0.d0)
mu(indx)=1.-!pi*p^2*iofr(c1,c2,c3,c4,z(indx),p)/norm
indx=where(z eq 0.d0)
if(total(indx) ge 0.) then mu(indx)=1.-!pi*p^2/norm
return
end

function iofr,c1,c2,c3,c4,r,p
sig1=sqrt(sqrt(1.-(r-p)^2))
sig2=sqrt(sqrt(1.-(r+p)^2))
return,1.-c1*(1.+(sig2^5-sig1^5)/5./p/r)-c2*(1.+(sig2^6-sig1^6)/6./p/r)$
  -c3*(1.+(sig2^7-sig1^7)/7./p/r)-c4*(p^2+r^2)
end


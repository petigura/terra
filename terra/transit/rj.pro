function rj,x,y,z,p
; This is a Numerical Recipes routine in FORTRAN modified for IDL
ERRTOL=.05
TINY=2.5e-13
BIG=9.E11
C1=3./14.
C2=1./3.
C3=3./22.
C4=3./26.
C5=.75*C3
C6=1.5*C4
C7=.5*C2
C8=C3+C3
rjj=dblarr(n_elements(x))
if(n_elements(x) ne n_elements(y) or n_elements(x) ne n_elements(z) or $
      n_elements(x) ne n_elements(p)) then begin
   print,'wrong size in rj'
   return,dblarr(n_elements(x))
endif
for i=0,n_elements(x)-1 do begin
  if(min([x(i),y(i),z(i)]) lt 0.d0 or min([x(i)+y(i),x(i)+z(i),y(i)+z(i),abs(p(i))]) lt TINY $
     or max([x(i),y(i),z(i),abs(p(i))]) gt BIG) then begin
      rjj(i) = 0.d0
  endif else begin
  sum=0.
  fac=1.
  if(p(i) gt 0 )then begin
    xt=x(i)
    yt=y(i)
    zt=z(i)
    pt=p(i)
  endif else begin
    xt=min([x(i),y(i),z(i)])
    zt=max([x(i),y(i),z(i)])
    yt=x(i)+y(i)+z(i)-xt-zt
    a=1./(yt-p(i))
    b=a*(zt-yt)*(yt-xt)
    pt=yt+b
    rho=xt*zt/yt
    tau=p(i)*pt/yt
    rcx=rc(rho,tau)
  endelse
LAB1:
  sqrtx=sqrt(xt)
  sqrty=sqrt(yt)
  sqrtz=sqrt(zt)
  alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
  alpha=(pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz)^2
  beta=pt*(pt+alamb)^2
  sum=sum+fac*rc(alpha,beta)
  fac=.25*fac
  xt =.25*(xt+alamb)
  yt =.25*(yt+alamb)
  zt =.25*(zt+alamb)
  pt =.25*(pt+alamb)
  ave=.2*(xt+yt+zt+pt+pt)
  delx=(ave-xt)/ave
  dely=(ave-yt)/ave
  delz=(ave-zt)/ave
  delp=(ave-pt)/ave
  if(max([abs(delx),abs(dely),abs(delz),abs(delp)]) gt ERRTOL)then goto,LAB1
  ea=delx*(dely+delz)+dely*delz
  eb=delx*dely*delz
  ec=delp^2
  ed=ea-3.*ec
  ee=eb+2.*delp*(ea-ec)
  rjj(i)=3.*sum+fac*(1.+ed*(-C1+C5*ed-C6*ee)+eb*(C7+delp*(-C8+delp*C4))$
      +delp*ea*(C2-delp*C3)-C2*delp*ec)/(ave*sqrt(ave))
  if (p(i) le 0.) then rjj(i)=a*(b*rjj(i)+3.*(rcx-rf(xt,yt,zt)))
  endelse
endfor
return,rjj
END

; Program for evaluating elliptic integrals: in notation of
; Gradsteyn and Rhyzik.
function rf,x,y,z
;double precision FUNCTION rf(x,y,z)
;implicit none
;double precision rf,x,y,z,ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4
ERRTOL=0.0025d0
;ERRTOL=1.d-5
TINY=1.5d-38
BIG=3.d37
THIRD=1.d0/3.d0
C1=1.d0/24.d0
C2=0.1d0
C3=3.d0/44.d0
C4=1.d0/14.d0
rff=dblarr(n_elements(x))
if(n_elements(x) ne n_elements(y) or n_elements(x) ne n_elements(z)) then begin
   print,'wrong size in rf'
   return,dblarr(n_elements(x))
endif
for i=0,n_elements(x)-1 do begin
  if(min([x(i),y(i),z(i)]) lt 0.d0 or min([x(i)+y(i),x(i)+z(i),y(i)+z(i)]) lt TINY $
     or max([x(i),y(i),z(i)]) gt BIG) then begin
      print,'invalid arguments in rf'
      print,min([x(i),y(i),z(i)]),min([x(i)+y(i),x(i)+z(i),y(i)+z(i)]),max([x(i),y(i),z(i)])
;      return,dblarr(n_elements(x))
      rff(i)=0.d0
  endif else begin
  xt=x(i)
  yt=y(i)
  zt=z(i)
LAB1:
  sqrtx=sqrt(xt)
  sqrty=sqrt(yt)
  sqrtz=sqrt(zt)
  alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
  xt=0.25d0*(xt+alamb)
  yt=0.25d0*(yt+alamb)
  zt=0.25d0*(zt+alamb)
  ave=THIRD*(xt+yt+zt)
  if(ave eq 0.d0) then begin
    delx=0.d0
    dely=0.d0
    delz=0.d0
  endif else begin
    delx=(ave-xt)/ave
    dely=(ave-yt)/ave
    delz=(ave-zt)/ave
  endelse
  if(max([abs(delx),abs(dely),abs(delz)]) gt ERRTOL) then goto,LAB1
  e2=delx*dely-delz*delz
  e3=delx*dely*delz
  rff(i)=(1.d0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave)
  endelse
endfor
return,rff
END

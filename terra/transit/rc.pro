function rc,x,y
;REAL rc,x,y,ERRTOL,TINY,SQRTNY,BIG,TNBG,COMP1,COMP2,THIRD,C1,C2,
;*C3,C4
ERRTOL=.04
;ERRTOL=1.d-3
TINY=1.69e-38
SQRTNY=1.3e-19
BIG=3.E37
TNBG=TINY*BIG
COMP1=2.236/SQRTNY
COMP2=TNBG*TNBG/25.
THIRD=1./3.
C1=.3
C2=1./7.
C3=.375
C4=9./22.
;REAL alamb,ave,s,w,xt,yt
rcc=dblarr(n_elements(x))
if(n_elements(x) ne n_elements(y)) then begin
   print,'wrong size in rc'
   return,dblarr(n_elements(x))
endif
for i=0,n_elements(x)-1 do begin
  if(x(i) lt 0. or y(i) eq 0. or (x(i)+abs(y(i))) lt TINY $
     or (x(i)*abs(y(i))) gt BIG or (y(i) lt -COMP1 and x(i) $
     gt 0. and x(i) lt COMP2)) then begin
      print,'invalid arguments in rc'
;      return,dblarr(n_elements(x))
     rcc(i)=0.d0
  endif else begin
;if(x.lt.0..or.y.eq.0..or.(x+abs(y)).lt.TINY.or.(x+
;     *abs(y)).gt.BIG.or.(y.lt.-COMP1.and.x.gt.0..and.x.lt.COMP2))pause 
;     *'invalid arguments in rc'
  if(y(i) gt 0.)then begin
    xt=x(i)
    yt=y(i)
    w=1.
  endif else begin
    xt=x(i)-y(i)
    yt=-y(i)
    w=sqrt(x(i))/sqrt(xt)
  endelse
LAB1:
  alamb=2.*sqrt(xt)*sqrt(yt)+yt
  xt=.25*(xt+alamb)
  yt=.25*(yt+alamb)
  ave=THIRD*(xt+yt+yt)
  s=(yt-ave)/ave
  if(abs(s) gt ERRTOL) then goto,LAB1
  rcc(i)=w*(1.+s*s*(C1+s*(C2+s*(C3+s*C4))))/sqrt(ave)
  endelse
endfor
return,rcc
END

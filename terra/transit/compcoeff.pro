;window,0,retain=2
.run occultnl
.run occultnl
.run occultsmall
.run occultsmall
!x.margin=[8,2]
!y.margin=[4,0]
!p.charthick=3
!p.thick=3
!x.thick=3
!y.thick=3
!p.charsize=1.
!x.charsize=1.0
!y.charsize=1.0
set_plot,'ps
device,filename='compcoeff.ps'
plotquery=0
b0=dindgen(401)*0.003d0
;b0=dindgen(41)*0.03d0
;!p.multi=[0,1,3]
;!y.margin=[-2,0]
; First, rl=0.01
;rl=0.01d0
;c1=1.d0
;c2=0.d0
;c3=0.d0
;c4=0.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;plot,b0,mulimb0,ys=1,yr=[0.9998d0,1.00001d0],ytitle='!4l!3',$
;   xtickname=[' ',' ',' ',' ',' ',' ',' ']
;oplot,b0,mulimbf,linestyle=1
;c1=0.d0
;c2=1.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;oplot,b0,mulimbf,linestyle=2
;c2=0.d0
;c3=1.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;oplot,b0,mulimbf,linestyle=3
;c3=0.d0
;c4=1.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;oplot,b0,mulimbf,linestyle=4
;xyouts,0.998,0.2,'p=0.1'
;;  Now a larger planet:
;!y.margin=[0,2]
rl=0.1d0
c1=1.d0
c2=0.d0
c3=0.d0
c4=0.d0
occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
plot,b0,mulimb0,ys=1,yr=[0.98d0,1.001d0],ytitle='!4l!3',xtitle='z'
;plot,b0,mulimb0,ys=1,yr=[0.99d0,1.001d0],ytitle='!4l!3',xtitle='z',xs=1,xr=[0.85,1.15]
;   xtickname=[' ',' ',' ',' ',' ',' ',' ']
oplot,b0,mulimbf,linestyle=1
occultsmall,rl,c1,c2,c3,c4,b0,mu
print,'c1 ',max(abs(mu-mulimbf))/max(1.-mulimbf)
oplot,b0,mu,color=150,linestyle=1
c1=0.d0
c2=1.d0
;c2=0.5d0
occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
oplot,b0,mulimbf,linestyle=2
occultsmall,rl,c1,c2,c3,c4,b0,mu
print,'c2 ',max(abs(mu-mulimbf))/max(1.-mulimbf)
oplot,b0,mu,color=150,linestyle=2
c2=0.d0
c3=1.d0
;c3=0.5d0
occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
oplot,b0,mulimbf,linestyle=3
occultsmall,rl,c1,c2,c3,c4,b0,mu
print,'c3 ',max(abs(mu-mulimbf))/max(1.-mulimbf)
oplot,b0,mu,color=150,linestyle=3
c3=0.d0
c4=1.d0
;c4=0.5d0
occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
oplot,b0,mulimbf,linestyle=4
occultsmall,rl,c1,c2,c3,c4,b0,mu
print,'c4 ',max(abs(mu-mulimbf))/max(1.-mulimbf)
oplot,b0,mu,color=150,linestyle=4
xyouts,0.998,0.2,'p=0.1'
;  Now an even larger planet:
;!y.margin=[4,0]
;rl=0.5d0
;c1=1.d0
;c2=0.d0
;c3=0.d0
;c4=0.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;plot,b0,mulimb0,ys=1,yr=[0.5d0,1.01d0],ytitle='!4l!3',xtitle='z'
;oplot,b0,mulimbf,linestyle=1
;c1=0.d0
;c2=1.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;oplot,b0,mulimbf,linestyle=2
;c2=0.d0
;c3=1.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;oplot,b0,mulimbf,linestyle=3
;c3=0.d0
;c4=1.d0
;occultnl,rl,c1,c2,c3,c4,b0,mulimbf,mulimb0,plotquery
;oplot,b0,mulimbf,linestyle=4
;xyouts,0.998,0.2,'p=0.1'
device,/close
set_plot,'x
command='gv compcoeff.ps &'
spawn,command

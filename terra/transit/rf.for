      FUNCTION rf(x,y,z)
      REAL*8 rf,x,y,z,ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4
      PARAMETER (ERRTOL=.08d0,TINY=1.5d-38,BIG=3.d37,THIRD=1.d0/3.d0,
     *C1=1.d0/24.d0,C2=.1d0,C3=3.d0/44.d0,C4=1.d0/14.d0)
      REAL*8 alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt
      if(min(x,y,z).lt.0.d0.or.min(x+y,x+z,y+z).lt.TINY.or.max(x,y,
     *z).gt.BIG)pause 'invalid arguments in rf'
      xt=x
      yt=y
      zt=z
1     continue
        sqrtx=sqrt(xt)
        sqrty=sqrt(yt)
        sqrtz=sqrt(zt)
        alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
        xt=.25d0*(xt+alamb)
        yt=.25d0*(yt+alamb)
        zt=.25d0*(zt+alamb)
        ave=THIRD*(xt+yt+zt)
        delx=(ave-xt)/ave
        dely=(ave-yt)/ave
        delz=(ave-zt)/ave
      if(max(abs(delx),abs(dely),abs(delz)).gt.ERRTOL)goto 1
      e2=delx*dely-delz**2
      e3=delx*dely*delz
      rf=(1.d0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave)
      return
      END
C  (C) Copr. 1986-92 Numerical Recipes Software 0NL&WR2.

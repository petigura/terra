      program testsmall
C Computes lightcurve for small planet approximation.
C Please cite Mandel & Agol (2002) if making use of this routine.
      implicit none
      integer i,nz
      parameter (nz=201)
      real*8 c1,c2,c3,c4,z(nz),mu(nz),p
c      write(6,*) 'Radius, limb-darkening coefficients'
      read(5,*) p,c1,c2,c3,c4
      do i=1,nz
        z(i)=2.d0*dble(i-1)/dble(nz-1)
      enddo
      call occultsmall(p,c1,c2,c3,c4,z,mu)
      do i=1,nz
        write(6,*) z(i),mu(i)
      enddo
      return
      end

      subroutine occultsmall(p,c1,c2,c3,c4,z,mu,nz)
      implicit none
      integer i,nz
      real*8 p,c1,c2,c3,c4,z(nz),mu(nz),i1,norm,
     &       x,tmp,iofr,pi
C This routine approximates the lightcurve for a small 
C planet. (See section 5 of Mandel & Agol (2002) for
C details):
C Input:
C  p      ratio of planet radius to stellar radius
C  c1-c4  non-linear limb-darkening coefficients
C  z      impact parameters (positive number normalized to stellar 
C        radius)- this is an array which MUST be input to the routine
C  NOTE:  nz must match the size of z & mu in calling routine
C Output:
C  mu     flux relative to unobscured source for each z
C
      pi=acos(-1.d0)
      norm=pi*(1.d0-c1/5.d0-c2/3.d0-3.d0*c3/7.d0-c4/2.d0)
      i1=1.d0-c1-c2-c3-c4
      do i=1,nz
        mu(i)=1.d0
        if(z(i).gt.1.d0-p.and.z(i).lt.1.d0+p) then
          x=1.d0-(z(i)-p)**2
          tmp=(1.d0-c1*(1.d0-0.8d0*x**0.25)
     &             -c2*(1.d0-2.d0/3.d0*x**0.5)
     &             -c3*(1.d0-4.d0/7.d0*x**0.75)
     &             -c4*(1.d0-0.5d0*x))
          mu(i)=1.d0-tmp*(p**2*acos((z(i)-1.d0)/p)
     &        -(z(i)-1.d0)*sqrt(p**2-(z(i)-1.d0)**2))/norm
        endif
        if(z(i).le.1.d0-p.and.z(i).ne.0.d0) then
          mu(i)=1.d0-pi*p**2*iofr(c1,c2,c3,c4,z(i),p)/norm
        endif
        if(z(i).eq.0.d0) then
          mu(i)=1.d0-pi*p**2/norm
        endif
      enddo
      return
      end
      
      function iofr(c1,c2,c3,c4,r,p)
      implicit none
      real*8 r,p,c1,c2,c3,c4,sig1,sig2,iofr
      sig1=sqrt(sqrt(1.d0-(r-p)**2))
      sig2=sqrt(sqrt(1.d0-(r+p)**2))
      iofr=1.d0-c1*(1.d0+(sig2**5-sig1**5)/5.d0/p/r)
     &         -c2*(1.d0+(sig2**6-sig1**6)/6.d0/p/r)
     &         -c3*(1.d0+(sig2**7-sig1**7)/7.d0/p/r)
     &         -c4*(p**2+r**2)
      return
      end

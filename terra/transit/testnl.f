      program testnl
      implicit none
      integer i,nb,j
      parameter (nb=10001)
      real*8 c1,c2,c3,c4,b0(nb),mulimb(nb,5),mulimb0(nb),rl
c      write(6,*) 'Radius, limb-darkening coefficients'
      read(5,*) rl,c1,c2,c3,c4
      do i=1,nb
        b0(i)=2.d0*dble(i-1)/dble(nb-1)
      enddo
      call occultnl(rl,c1,c2,c3,c4,b0,mulimb0,mulimb,nb)
      do i=1,nb
        write(6,*) b0(i),mulimb0(i),(mulimb(i,j),j=1,5)
      enddo
      return
      end

      program test
      implicit none
      integer i,nz
      parameter(nz=101)
      double precision p,u1,u2,muo1(nz),mu0(nz),z(nz)
      write(6,*) 'p, u1, u2 '
      read(5,*) p,u1,u2
      do i=1,nz
        z(i)=(dble(i)-1.d0)/dble(nz)*2.d0
      enddo
      call occultquad(z,u1,u2,p,muo1,mu0,nz)
      do i=1,nz
        write(6,*) z(i),muo1(i),mu0(i)
      enddo
      return
      end

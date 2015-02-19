      function ellec(k)
      implicit none
      double precision k,m1,a1,a2,a3,a4,b1,b2,b3,b4,ee1,ee2,ellec
C Computes polynomial approximation for the complete elliptic
C integral of the second kind (Hasting's approximation):
      m1=1.d0-k*k
      a1=0.44325141463d0
      a2=0.06260601220d0
      a3=0.04757383546d0
      a4=0.01736506451d0
      b1=0.24998368310d0
      b2=0.09200180037d0
      b3=0.04069697526d0
      b4=0.00526449639d0
      ee1=1.d0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
      ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*log(1.d0/m1)
      ellec=ee1+ee2
      return
      end

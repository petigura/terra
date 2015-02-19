function ellk,k
; Computes polynomial approximation for the complete elliptic
; integral of the first kind (Hasting's approximation):
m1=1.d0-k^2
a0=1.38629436112d0
a1=0.09666344259d0
a2=0.03590092383d0
a3=0.03742563713d0
a4=0.01451196212d0
b0=0.5d0
b1=0.12498593597d0
b2=0.06880248576d0
b3=0.03328355346d0
b4=0.00441787012d0
ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*alog(m1)
return,ek1-ek2
end

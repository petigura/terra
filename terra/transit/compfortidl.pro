command='f77 -c -O3 testnl.f -o testnl.o'
spawn,command
command='f77 -c -O3 occultnl.f -o occultnl.o'
spawn,command
command='f77 -O3 testnl.o occultnl.o -o testnl'
spawn,command
command='rm test.out'
spawn,command
timing=systime(1)
command='./testnl < test.in > test.out '
spawn,command
print,'Fortran ',systime(1)-timing
openr,1,'test.in'
b=fltarr(5)
readf,1,b
close,1
b0=2.d0*dindgen(10001)/10000.
.run occultnl
.run occultnl
timing=systime(1)
occultnl,b(0),b(1),b(2),b(3),b(4),b0,mulimb0,mulimbf,0
print,'IDL     ',systime(1)-timing
openr,1,'test.out'
a=fltarr(7,2001)
readf,1,a
close,1
print,max(abs(mulimb0-a(1,*)))

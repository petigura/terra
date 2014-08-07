c
c
      subroutine bls(n,t,x,u,v,nf,fmin,df,nb,qmi,qma,
     &p,bper,bpow,depth,qtran,in1,in2)
c
c------------------------------------------------------------------------
c     >>>>>>>>>>>> This routine computes BLS spectrum <<<<<<<<<<<<<<
c
c        [ see Kovacs, Zucker & Mazeh 2002, A&A, Vol. 391, 369 ]
c------------------------------------------------------------------------
c
c     Input parameters:
c     ~~~~~~~~~~~~~~~~~
c
c     n    = number of data points
c     t    = array {t(i)}, containing the time values of the time series
c     x    = array {x(i)}, containing the data values of the time series
c     u    = temporal/work/dummy array, must be dimensioned in the 
c            calling program in the same way as  {t(i)}
c     v    = the same as  {u(i)}
c     nf   = number of frequency points in which the spectrum is computed
c     fmin = minimum frequency (MUST be > 0)
c     df   = frequency step
c     nb   = number of bins in the folded time series at any test period       
c     qmi  = minimum fractional transit length to be tested
c     qma  = maximum fractional transit length to be tested
c
c     Output parameters:
c     ~~~~~~~~~~~~~~~~~~
c
c     p    = array {p(i)}, containing the values of the BLS spectrum
c            at the i-th frequency value -- the frequency values are 
c            computed as  f = fmin + (i-1)*df
c     bper = period at the highest peak in the frequency spectrum
c     bpow = value of {p(i)} at the highest peak
c     depth= depth of the transit at   *bper*
c     qtran= fractional transit length  [ T_transit/bper ]
c     in1  = bin index at the start of the transit [ 0 < in1 < nb+1 ]
c     in2  = bin index at the end   of the transit [ 0 < in2 < nb+1 ]
c
c
c     Remarks:
c     ~~~~~~~~ 
c
c     -- *fmin* MUST be greater than  *1/total time span* 
c     -- *nb*   MUST be lower than  *nbmax* 
c     -- Dimensions of arrays {y(i)} and {ibi(i)} MUST be greater than 
c        or equal to  *nbmax*. 
c     -- The lowest number of points allowed in a single bin is equal 
c        to   MAX(minbin,qmi*N),  where   *qmi*  is the minimum transit 
c        length/trial period,   *N*  is the total number of data points,  
c        *minbin*  is the preset minimum number of the data points per 
c        bin.
c     
c========================================================================
c
      implicit real*8 (a-h,o-z)
c
      dimension t(*),x(*),u(*),v(*),p(*)
      dimension y(20000),ibi(20000)
c
      minbin = 5
      nbmax  = 20000
c     Number of bins specified by the user cannot exceed hard-coded amount
      if(nb.gt.nbmax) write(*,*) ' HERE NB > NBMAX !!'
      if(nb.gt.nbmax) stop
c     tot is the time baseline
      tot=t(n)-t(1)
c     The minimum frequency must be greater than 1/T
c     We require there be one dip.
c     To nail down the period, wouldn't we want 2 dips.
      if(fmin.lt.1.0d0/tot) write(*,*) ' fmin < 1/T !!'
      if(fmin.lt.1.0d0/tot) stop
c------------------------------------------------------------------------
c
c     turn n into a float
      rn=dfloat(n)

c     integer ceiling of the product
c     bin number of binned points in transit
      kmi=idint(qmi*dfloat(nb))

      if(kmi.lt.1) kmi=1
c     maximum number of binned points in transit
      kma=idint(qma*dfloat(nb))+1
c     minimum number of (unbinned) points in transit
      kkmi=idint(rn*qmi)
      if(kkmi.lt.minbin) kkmi=minbin
      bpow=0.0d0
c
c=================================
c     Set temporal time series
c=================================
c
      s=0.0d0
      t1=t(1)
c     A simple for loop, the `103' is the close paren for the loop.
      do 103 i=1,n
c     Time of measument `i' relative to start
      u(i)=t(i)-t1
c     s is the sum of the measurements 
      s=s+x(i)
 103  continue
c     s is now the average measurement value
      s=s/rn
      do 109 i=1,n
c     v is measument array with the mean subtracted
      v(i)=x(i)-s
 109  continue
c
c******************************
c     Start period search     *
c******************************
c
      do 100 jf=1,nf
c     jf - index for stepping through trial frequencies
      f0=fmin+df*dfloat(jf-1)
c     p0 - period corresponding to the trial frequency
      p0=1.0d0/f0
c
c======================================================
c     Compute folded time series with  *p0*  period
c======================================================
c
      do 101 j=1,nb
c     zeroing out columns of y and ibi (their length is 10000)
        y(j) = 0.0d0
      ibi(j) = 0
 101  continue
c
      do 102 i=1,n  
c     ph is the phase of the ith measurement (note their is a 2pi
c     missing)
      ph     = u(i)*f0
c     modulo(phase,period)
      ph     = ph-idint(ph)
c     j what bin is the ith data point in?
      j      = 1 + idint(nb*ph)
c     ibi counts the number of points in the bin
      ibi(j) = ibi(j) + 1
c     y sums the points 
       y(j)  =   y(j) + v(i)
 102  continue   
c
c===============================================
c     Compute BLS statistics for this period
c===============================================
c
      power=0.0d0
c     i now is now an index corresponding to the start of transit
c     double nested loop
      do 1 i=1,nb
      s     = 0.0d0
      k     = 0
      kk    = 0
c     bin corresponding to end of transit
      nb2   = i+kma
c     end of transit cannot be > nb
      if(nb2.gt.nb) nb2=nb
c     scan in transit width.
      do 2 j=i,nb2
c     increase counter by 1
      k     = k+1
c     increase kk by the number of points in the bin
      kk    = kk+ibi(j)
      s     = s+y(j)
c     Do not compute BLS if the number binned or unbinded points is less
c     than the predetermined amount
      if(k.lt.kmi) go to 2
      if(kk.lt.kkmi) go to 2
c     rn1 - number of unbinned points in current trial width
c     rn  - number of toralk unbinned points
      rn1   = dfloat(kk)
      pow   = s*s/(rn1*(rn-rn1))
      if(pow.lt.power) go to 2
c     power stores the maximum power.
      power = pow
      jn1   = i
      jn2   = j
      rn3   = rn1
      s3    = s
 2    continue
 1    continue
c
      power = dsqrt(power)
      p(jf) = power
c
      if(power.lt.bpow) go to 100
      bpow  =  power
      in1   =  jn1
      in2   =  jn2
      qtran =  rn3/rn
      depth = -s3*rn/(rn3*(rn-rn3))
      bper  =  p0
c
 100  continue
c
      return
      end
c
c

all:
	cd terra/FFA/ && python setup.py build_ext --inplace
	cd terra/transit/ && f2py -c occultsmall.f -m occultsmall


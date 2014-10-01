# Shell script to facilitate creation of batch jobs for TERRA.
#!/usr/bin/env bash


echo ""
echo "Setting up batch scripts"
echo ""

PARDB=$1 # Name of the parameter database
TPSDIR=${HOME}/TPS/C0
PHOTDIR=${K2_DIR}/photometry/C0/
K2_SCRIPTS=${K2_DIR}/code/py/scripts

#read -ep "Enter output directory " TPSDIR
SCRIPTSDIR=${TPSDIR}/scripts
OUTPUTDIR=${TPSDIR}/output

mkdir -p ${TPSDIR}
mkdir -p ${SCRIPTSDIR}
mkdir -p ${OUTPUTDIR}

#read -ep "Enter photometry directory" PHOTDIR

# Generate a list of the EPIC IDs
find ${PHOTDIR} -name "*.fits" |
awk -F "/" '{print $(NF)}' | # Grab the file basename
awk -F "." '{print $1}' | # Hack off the .fits part
sort > photfiles.temp # sorting is necessary for join

# Generate a list of the stars we wish to analyze
sqlite3 -noheader ${PARDB} "select epic from pp sort" | 
awk '{print $1}' | 
tail -n +2 > epiclist.temp

# Figure out how many of the requested stars have extant photometry
join photfiles.temp epiclist.temp > phot_epic_join.temp
N_PHOT_EXTANT=$( cat phot_epic_join.temp | wc -l)
N_PARS=$(cat epiclist  | wc -l )

echo "Photometry exists for ${N_PARS} out of ${N_PHOT_EXTANT} stars"
for epic in `cat phot_epic_join.temp`
do
    STAROUTPUTDIR=${OUTPUTDIR}/${epic}
    GRIDFILE=${STAROUTPUTDIR}/${epic}.grid.h5
    FITSFILE=${PHOTDIR}/${epic}.fits
    echo "# TERRA #"
    echo ". $HOME/k2_setup.sh"
    echo "cd $K2_DIR"
    echo "mkdir -p ${STAROUTPUTDIR}"
    echo "python ${K2_SCRIPTS}/terraWrap.py pp ${FITSFILE} ${GRIDFILE} ${PARDB} ${epic}"
    echo "python ${K2_SCRIPTS}/terraWrap.py grid ${GRIDFILE} ${PARDB} ${epic}"
    echo "python ${K2_SCRIPTS}/terraWrap.py dv ${GRIDFILE} ${PARDB} ${epic}"
    echo "chmod -R o+rX ${STAROUTPUTDIR}"
done > ${SCRIPTSDIR}/terra.tot

rm photfiles.temp epiclist.temp phot_epic_join.temp 

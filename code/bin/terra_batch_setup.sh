#!/usr/bin/env bash
echo Please enter name of .obs file:
echo ---------format---------
echo CK01299 rj116.663
echo CK00372 rj116.664
echo CK00976 rj117.149
read -e OBSFILE
echo ""

if [ -e "$OBSFILE" ] 
then
    NLINES=$( cat "$OBSFILE" | wc -l )
    echo "$OBSFILE has $NLINES spectra" 
else
    echo "$OBSFILE does not exist!" 
    exit 1
fi

BNAME=$(basename ${OBSFILE%.obs})

read -ep "Enter output directory " SM_PROJDIR

if [ -d "$SM_PROJDIR" ] 
then
    echo "Directory exists, could be overwriting files" 
    exit 1
else
    H5DIR="${SM_PROJDIR}/output/h5/"
    PLOTDIR="${SM_PROJDIR}/output/plots/"
    SCRIPTDIR="${SM_PROJDIR}/scripts/"
    
    mkdir -p ${H5DIR} ${PLOTDIR} ${SCRIPTDIR}

    echo "h5 structures will be stored here: ${H5DIR}"
    echo "plots will be stored here: ${PLOTDIR}"
    echo "scripts will be stored here: ${SCRIPTDIR}"
fi

# Check that all of the desired spectra have been put on rest WlS.
DB_EXTANT_LIST="${BNAME}_extant_db.temp"
DB_ANALYZE_LIST="${BNAME}_analyze_db.temp"
while read -r name obs; do
    echo "${obs}.fits"
done < ${OBSFILE} | sort > ${DB_ANALYZE_LIST}

find $SM_DIR/spectra/iodfitsdb/ -name "*.fits" |  awk -F 'rj' '{print "rj"$2}' | sort > ${DB_EXTANT_LIST}
N_DB_EXTANT=$(join  ${DB_EXTANT_LIST} ${DB_ANALYZE_LIST} | wc -l)

if [ ${N_DB_EXTANT} -ne $NLINES ]
then
    echo "Only have ${N_DB_EXTANT} of ${NLINES} "
    exit 1
fi

# # Check that all of the desired spectra have been put on rest WlS.
# SPEC_EXTANT_LIST="${BNAME}_extant_restwav.temp"
# SPEC_ANALYZE_LIST="${BNAME}_analyze_restwav.temp"
# while read -r name obs; do
#     echo "${name}_${obs}.h5"
# done < ${OBSFILE} > ${SPEC_ANALYZE_LIST}
# 
# ls $SM_DIR/spectra/restwav/*.h5 | xargs -n1 basename > ${SPEC_EXTANT_LIST}
# N_SPEC_EXTANT=$(join  ${SPEC_EXTANT_LIST} ${SPEC_ANALYZE_LIST} | wc -l)
# 
# if [ ${N_SPEC_EXTANT} -ne $NLINES ]
# then
#     echo "Only have ${N_SPEC_EXTANT} of ${NLINES} spectra on restwav length"
#     exit 1
# fi


# Run in debug mode?
read -p "Run in debug mode? [y/n] " yn
case $yn in
    [Yy]* ) DEBUG='--debug';;
    [Nn]* ) DEBUG='';;
    * ) echo "Please answer yes or no.";;
esac

read -p "Run ncores? " ncores
if [ ${ncores} -ne 1 ]
then
    NCORES="--np=${ncores}"
else
    NCORES=""
fi

while read -r name obs; do
    echo ". $HOME/sm_setup.sh"
    echo "cd $SM_DIR"
    echo "export SM_PROJDIR=$SM_PROJDIR"
    echo "python code/py/scripts/restwav_batch.py -f --obs ${obs}"
    echo "python code/py/scripts/specmatch_batch.py ${obs} ${DEBUG} ${NCORES}"
    echo "python code/py/scripts/add_telluric.py ${obs} --plot"
    echo "python code/py/scripts/add_polish.py ${obs} fm"
    echo "python code/py/scripts/smplots_batch.py --panels --matches-chi --polish --quicklook ${obs}"
done < ${OBSFILE} > ${SCRIPTDIR}/${BNAME}.tot

rm ${BNAME}*temp



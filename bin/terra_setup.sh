# Shell script to facilitate creation of batch jobs for TERRA.
#!/usr/bin/env bash

echo "Script name: $0"
if [ "$#" -ne 1 ]; 
    then echo "signature: terra_setup.sh epic_list"
    exit 1
fi

echo ""
echo "Setting up batch scripts"
echo ""

EPICLIST=$1 # Name of the parameter database
K2_SCRIPTS=${K2_DIR}/code/py/scripts

read -ep "Enter output directory: " TPSDIR
SCRIPTSDIR=${TPSDIR}/scripts
OUTPUTDIR=${TPSDIR}/output

echo mkdir -p ${TPSDIR}
echo mkdir -p ${SCRIPTSDIR}
echo mkdir -p ${OUTPUTDIR}

mkdir -p ${TPSDIR}
mkdir -p ${SCRIPTSDIR}
mkdir -p ${OUTPUTDIR}

read -ep "Enter photometry directory: " PHOTDIR

# Generate a list of the EPIC IDs
#find ${PHOTDIR} -name "*.fits" |
#awk -F "/" '{print $(NF)}' | # Grab the file basename
#awk -F "." '{print $1}' | # Hack off the .fits part
#sort > photfiles.temp # sorting is necessary for join
#
#
## Figure out how many of the requested stars have extant photometry
#join photfiles.temp ${EPICLIST} > phot_epic_join.temp
#N_PHOT_EXTANT=$( cat phot_epic_join.temp | wc -l)
#N_PARS=$(cat ${EPICLIST}  | wc -l )
#
#echo "Photometry exists for ${N_PARS} out of ${N_PHOT_EXTANT} stars"

while read epic
do
    echo "${K2_DIR}/code/bin/terra.sh ${epic}"
done < ${EPICLIST} > ${SCRIPTSDIR}/terra.tot

#rm photfiles.temp epiclist.temp phot_epic_join.temp 

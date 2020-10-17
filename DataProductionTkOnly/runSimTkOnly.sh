#!/bin/bash

# a helper function to parse PU values
function list_include_item {
    local list="$1"
    local item="$2"
    if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]])  ]] ; then
        # yes, list include item
        result=0
    else
        result=1
    fi
    return $result
}

#assign the command line arguments
PU=$1
NEVENTS=$2
JOBID=$3
STAGED=$4

################################################################################
##CHANGE ME ACCORDING TO YOUR NEEDS
################################################################################
NTHREADS=10

FILE=file:
PUPATH=/afs/cern.ch/work/g/gauzinge/public/minBiasFiles
#PUPATH=/afs/cern.ch/work/g/gauzinge/public

#OUTDIR=/eos/user/g/gauzinge/PUdata
#OUTDIR=.
#
# COB high-pileuo samples
#OUTDIR=/eos/user/c/cbarrera/BRIL_ITsim_PUsamples
#
#OUTDIR=/afs/cern.ch/work/c/cbarrera/private/BRIL/outputDir
OUTDIR=/afs/cern.ch/user/g/gauzinge/BIBSim/CMSSW_11_2_0_pre6/src/BRIL_ITsim

#additional variables for mixing module
BUNCHSPACING=25
MINBUNCH=-12
MAXBUNCH=3
################################################################################
################################################################################
################################################################################

#make the list of minbias files and remove trailing comma
for filename in ${PUPATH}/*TkOnly.root; do
    FILENAMES+=${FILE}${filename},
done
PUFILE=${FILENAMES%,}

#some sanity checks on the command line arguments
if test -z "$PU" 
then
   echo "No Pileup number specified - please run as source runSim.sh PU"
   return
fi

if `list_include_item "0 0.5 1 1.5 2 10 20 25 30 35 40 45 50 70 75 100 125 140 150 175 200" $PU` ; then
  #echo "PU value $PU available in list"
  echo ''
else 
  echo "ERROR, not a valid PU value! quitting!"
  return
fi

if test -z "$NEVENTS" 
then
   echo "No # of Events specified. Please run as runSim.sh PU NEVENTS if you want to use a number different than default"
   echo "The default value is 10 events"
   NEVENTS=10
   #return
fi

if test -z "$JOBID" 
then
   echo "No job ID passed - I will exit"
   return
fi

################################################################################
#PRINT THE ARGUMENTS SUMMARY
################################################################################

echo '###########################################################################'
echo 'Configuration: '
echo 'Pileup Average: '${PU}
echo 'Number of Events: '${NEVENTS}
echo 'JobId: '${JOBID}
echo 'Bunchspace: ' ${BUNCHSPACING}
echo 'minBunch :' ${MINBUNCH}
echo 'maxBunch : ' ${MAXBUNCH}
echo 'PileupFiles: ' ${PUFILE}
echo 'OutputDirectory: ' ${OUTDIR}
echo 'Number of Threads: ' ${NTHREADS}
echo '###########################################################################'

################################################################################
#SETUP CMSSW FRAMEWORK
################################################################################

#Extract sandbox
tar -xf sandbox.tar.bz2
#Keep track of release sandbox version
basedir=$PWD
rel=$(echo CMSSW_*)
arch=$(ls $rel/.SCRAM/|grep slc) || echo "Failed to determine SL release!"
old_release_top=$(awk -F= '/RELEASETOP/ {print $2}' $rel/.SCRAM/slc*/Environment) || echo "Failed to determine old releasetop!"
 
# Creating new release
# This isdone so e.g CMSSW_BASE and other variables are not hardcoded to the sandbox setting paths
# which will not exist here
  
echo ">>> creating new release $rel"
mkdir tmp
cd tmp
scramv1 project -f CMSSW $rel
new_release_top=$(awk -F= '/RELEASETOP/ {print $2}' $rel/.SCRAM/slc*/Environment)
cd $rel
echo ">>> preparing sandbox release $rel"
  
#for i in bin cfipython config lib module python src; do
for i in bin cfipython config lib python src; do
    rm -rf "$i"
    mv "$basedir/$rel/$i" .
done
     
echo ">>> fixing python paths"
for f in $(find -iname __init__.py); do
   sed -i -e "s@$old_release_top@$new_release_top@" "$f"
done
         
eval $(scramv1 runtime -sh) || echo "The command 'cmsenv' failed!"
cd "$basedir"
echo "[$(date '+%F %T')] wrapper ready"


################################################################################
##RUN THE ACTUAL SIMULATION
################################################################################

echo "Running the full simulation in one step from directory ${PWD}!"
command="cmsRun python/BRIL_ITsimPUTkOnly_cfg.py print \
        nEvents=${NEVENTS} \
        pileupFile=${PUFILE} \
        pileupAverage=${PU} \
        bunchSpace=${BUNCHSPACING} \
        minBunch=${MINBUNCH} \
        maxBunch=${MAXBUNCH} \
        nThreads=${NTHREADS} \
        jobId=${JOBID} \
        outputDirectory=file:${OUTDIR}"

echo 'Command: ' ${command}
${command}

################################################################################
##CLEANING UP BEHIND MYSELF
################################################################################

echo "Done running the generation"
echo "Cleaning up behing me"
rm -rf tmp/
rm -rf CMSSW_11_2_0_pre6/


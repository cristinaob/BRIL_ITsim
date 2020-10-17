#!/bin/bash

#assign the command line arguments
NEVENTS=$1
JOBID=$2

################################################################################
##CHANGE ME ACCORDING TO YOUR NEEDS
################################################################################
NTHREADS=10
OUTDIR=/afs/cern.ch/work/g/gauzinge/public/minBiasFiles
#OUTDIR=/afs/cern.ch/work/g/gauzinge/public/
################################################################################
################################################################################
################################################################################

#some sanity checks on the command line arguments
if test -z "$NEVENTS" 
then
   echo "No # of Events specified. Please run as runSim.sh PU NEVENTS if you want to use a number different than default"
   #echo "The default value is 10 events"
   #NEVENTS=10
   return
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
echo 'Number of Events: '${NEVENTS}
echo 'JobId: '${JOBID}
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

echo "Running the full minimum Bias generation from directory ${PWD}!"
command="cmsRun python/BRIL_ITsimMinBiasTkOnly_cfg.py print \
            nEvents=${NEVENTS} \
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

# The _not yet_ ultimate guide to BRIL Inner Tracker Simulations

### Preface
This guide is intended for people working on the BRIL Phase II upgrade, more specifically simulations of the Inner Tracker for Lumi measurements. It gives detailed instructions on how to set up and run simulations with custom IT geometries and varying levels of pileup. It is sectioned in the following parts:

1. setting up an appropriate CMSSW Environment
1. using custom geometry files exported from [TkLayout](http://tklayout.web.cern.ch)
1. generating Minimum Bias Events using the custom geometry to use as Pileup input for the rest of the simulation
1. running the full simulation
    * locally for a small data sample using the runTheMatrix.py command of CMSSW
    * locally using the runSim.sh script that wraps the above for more convenience
    * on the HT Condor batch system using the runSim.sh script and a submission file
1. runnig the BRIL ITclusterAnalyzer on the output EDM file to populate BRIL relevant histograms for further analysis

### Setting up an Environment
It is preferrable to run these steps on a machine with access to the CERN cvmfs read-only filesystem as this allows to use different CMSSW releases -- this usually means lxplus. Before you get started, make sure that you have a few GB of available space in the directory you want to run in. This could either be /afs or /eos. The release this guide is based on is `CMSSW_10_4_0_pre2` that you set up like so:

```sh
source $VO_CMS_SW_DIR/cmsset_default.sh
mkdir mySimDir
cd mySimDir
cmsrel CMSSW_10_4_0_pre2
cd CMSSW_10_4_0_pre2/src/
```

This creates a CMSSW working area in the directory mySimDir/ where a directory tree is created in the CMSSW_10_4_0_pre2/ folder. All your code and packages are always going to the `/src` subdirecory - it is the only place where you will be working.

```sh
cmsenv
```

This sets up the working environment, the paths and the right compiler. In the next step we need to check out some packages from the CMSSW github repo that we want to modify. Specifically, these will contain the modified geometry files. 

```sh
git cms-addpkg Configuration/PyReleaseValidation
git cms-addpkg Geometry/TrackerCommonData
git cms-addpkg Geometry/TrackerRecoData
git cms-addpkg Geometry/TrackerSimData
git cms-addpkg SLHCUpgradeSimulations/Geometry
```

### Using custom Geometries
Once done with the previous step, you have to get the custom geometry files and put them in the appropriate place. First, you have to check out this repo in your CMSSW work area, so `mySimDir/CMSSW_10_4_0_pre2/src`.

```sh
cd mySimDir/CMSSW_10_4_0_pre2/src
git clone https://github.com/gauzinge/BRIL_ITsim.git
cd BRIL_ITsim
```

or better yet fork the repo to your github account! Now you can call the `copyGeo.sh` with the following arguments: `-s mysourceDir -d mydestDir` to copy all relevant geometry files from the source (most likely `BRIL_ITsim/ITGeometries/OT614_200_IT612` in this repo as source and `mySimDir/CMSSW_10_4_0_pre2/src` as destination). 

```sh
source copyGeo.sh -s ITGeometries/OT614_200_IT612 -d mySimDir/CMSSW_10_4_0_pre2/src
```

The next step is crucial, so pay attention: since you modified the geometry files you have to re-build your added CMSSW sources. This is done using the scram command

```sh
scram b -j8
```

for 8 core compilation. Now everything you do will be based on the custom geometry.

### Generating Minimum Bias Events to use as Pileup Input in the actual Simulation

The next step is to generate a large enough sample of Minimum Bias events for your scenario that can later be used as Pileup input for the actual simulation. To do this, you need to run the first step of the workflow generated by the CMS `runTheMatrix.py` command. It is important to choose the right scenario which in our case is the Phase II upgrade. The name is *2023D21*.

```sh
runTheMatrix.py --what upgrade -l 21240 -ne
```

will just dump the commands to your terminal but I suggest running the actual cmsDriver command instead:

```sh
cmsDriver.py MinBias_14TeV_pythia8_TuneCUETP8M1_cfi  --conditions auto:phase2_realistic -n 2000 --era Phase2 --eventcontent FEVTDEBUG --relval 90000,100 -s GEN,SIM --datatier GEN-SIM --beamspot HLLHC14TeV --geometry Extended2023D21 --nThreads 4
```

this generates 2000 Minimum Bias Events in a file called ` MinBias_14TeV_pythia8_TuneCUETP8M1_cfi_GEN_SIM.root` -- put it somewhere where you won't accidentially delete it:

```sh
cd mySimDir/CMSSW_10_4_0_pre2/src
mkdir myMinBiasSample
mv  MinBias_14TeV_pythia8_TuneCUETP8M1_cfi_GEN_SIM.* myMinBiasSample/
```

Congratulations, you are almost done! In order to use these events you have to edit a file in `mySimDir/CMSSW_10_4_0_pre2/src/Configuration/PyReleaseValidation/python/relval_steps.py`:
the easiest is to search for the string _PUData_ in your editor. Then edit these lines:

```python
PUDataSets={}
for ds in defaultDataSets:
    key='MinBias_14TeV_pythia8_TuneCUETP8M1'+'_'+ds
    name=baseDataSetReleaseBetter[key]
#    if '2017' in name or '2018' in name:
    if '2017' in name:
    	PUDataSets[ds]={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(name,)}
    elif '2018' in name:
    	PUDataSets[ds]={'-n':10,'--pileup':'AVE_50_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(name,)}
    else:
        PUDataSets[ds]={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_14TeV/%s/GEN-SIM'%(name,)}
```

change the last line to point to your Minimum Bias Sample file (be sure to use an absolute path from your home direcotory - otherwise batched simulations won't work):

```python
PUDataSets[ds]={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'file:mySimDir/CMSSW_10_4_0_pre2/src/myMinBiasSample/MinBias_14TeV_pythia8_TuneCUETP8M1_cfi_GEN_SIM.root'}
```

this will tell CMSSW to use your Minimum Bias Data sample as Pileup input instead of some files with a standard geometry from the database. This is really important. Next, you have to once more compile your work environment to make it pick up the changes:

```sh
scram b -j8
```

Congratulations, you are all set for running the actual simulation!

### Running the full Simulation
There are 3 possible ways to run the full simulation. The normal way using the `runTheMatrix.py` command, a wrapper script called `runSim.sh` that is part of this repo or in batch mode on the CERN HT Condor batch service. The details will be described in the following sections!

#### RunTheMatrix
Now, to test your working environment, you can use the `runTheMatrx.py` script provided with CMSSW. Again, this will launch a 3 step process of generating events of a certain type, simulating the detector response and reconstruction the events. More specifically, the pixel clustering happens in step3 (the RECO step). The scenario we want to use for our purposes is a single Neutrino (so an empty detector) overlaid with a variable number of pileup. The scenario has the label **21461**. So in your CMMSW `src` directory run:

```sh
runTheMatrix.py --what upgrade -l 21461 --command "-n 100"
```

The `--command "-n 100"` specifies 100 events. Change it if you want but be patient. Oh, and the default pileup number is 35. You can't easily change it in this workflow but it is essentially just a test to test the environment. Wait for the process to complete and then check your working directory: you should see files step1.root, step2.root and step3.root. The file you are interested in is **always** `step3.root`. Try verifying that it contains a collection of type *SiPixelCluster*:

```sh
edmDumpEventContent step3.root | grep SiPixelCluster
```

See something? Great, you are done for this part. 

#### Running simulations locally using the runSim.sh script
Since the above workflow is tedious and does not really give you fine control over the options (plus it runs a bunch of processes that we don't need) it is usually better to use the `runSim.sh` script provided with this repo. It has some filepaths hardcoded (for example the path to put the output files and the pileup input file) so you want to open it in your editor and fix all the paths at the top of the script - you may also want to change the number of threads to something reasonable for your job. 

Before you continue however, you need to sandbox your CMSSW working environment. This is important for batch processing but we'll also use it for this case. To do that, create a new direcotry somewhere on your /afs or /eos and copy the CMSSW_10_4_0_pre2 direcotry there. Alternatively you can just use the sandbox provided with the repo but it's the geometry mentioned above so be prudent. You'll need to repeat the steps below for each new geometry.

```sh
cd
mkdir test
cp -rf mySimDir/CMSSW_10_4_0_pre2/ test
```

next, cd to that directory and remove all unnecessary files (don't be shy, you are working on a copy).

```sh
cd ~/test/CMSSW_10_4_0_pre2/src/
rm -rf BRIL_ITsim
rm -rf myMinBiasSample
```

Next, tar up your sandbox and copy it to your working area `mySimDir/CMSSW_10_4_0_pre2/src/BRIL_ITsim/`:

```sh
cd test
tar -jcf sandbox.tar.bz2 CMSSW_10_4_0_pre2/
cp sandbox.tar.bz2 mySimDir/CMSSW_10_4_0_pre2/src/BRIL_ITsim
```

Now you are ready to run the script. The `runSim.sh` script takes 3 command line arguments

```sh
./runSim.sh PU NEVENTS JOBID
```

The first one is the PU you want to run and the list of possible values can be found in the script, the second one is the number of events and the third one is a jobid (irrelevant for this case but important in batch mode) - please always provide all three. For now you can set the jobid to 0. This runs all three steps of the `runTheMatrix.py` command and in addition slimms the output *step3.root* file down to only conatain pixel relevant collections. The resulting output file is called `step3_pixel_PU_${PU}.${JOBID}.root` where ${PU} and ${JOBID} are replaced with your command line arguments. This is the file we will use as input to the ITclusterAnalyzer later.

#### Runnig simulations on CERN HT Condor batch service

Before you do anything, first read the HT Condor [guide](http://batchdocs.web.cern.ch/batchdocs/tutorial/introduction.html) including chapter 3. 



Done? Good, now you know about submission files. Have a look at the `PUgeneration.sub` file proveded with this repo. Change the PU parameter variable, the NEvents variable, the request_cpu number to match your number of threads in the `runSim.sh` script and the number of parallel jobs. Try with a small number of events and only one job first to verify that things work ok.

```sh
condor_submit PUgeneration.sub
```

and be patient and watch as your quota goes away....Happy simulating!


###  Running the BRIL IT Cluster analyzer

coming soon ...




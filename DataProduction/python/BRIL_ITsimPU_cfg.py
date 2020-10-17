# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
##STEP1
# with command line options: SingleNuE10_cfi.py --mc --conditions auto:phase2_realistic -n ${NEVENTS} --era Phase2 --eventcontent ${EVENTCONTENT} --relval 25000,250 -s GEN,SIM --datatier GEN-SIM --beamspot HLLHC --geometry Extended2023D21 --fileout file:step1.root --nThreads ${NTHREADS}
##STEP2
# with command line options: step2 --mc --conditions auto:phase2_realistic --pileup_input file:${PUFILE} -n ${NEVENTS} --era Phase2 --eventcontent ${EVENTCONTENT} -s DIGI:pdigi_valid,L1,DIGI2RAW --datatier GEN-SIM-DIGI-RAW --pileup ${PUSTRING} --geometry Extended2023D21 --filein file:step1.root --fileout file:step2.root --nThreads ${NTHREADS}
##STEP3
# with command line options: step3 --mc --conditions auto:phase2_realistic --pileup_input file:${PUFILE} -n ${NEVENTS} --era Phase2 --eventcontent ${EVENTCONTENT} --runUnscheduled -s RAW2DIGI,L1Reco,RECO,RECOSIM --datatier GEN-SIM-RECO --pileup ${PUSTRING} --geometry Extended2023D21 --filein file:step2.root --fileout file:step3_pixel_PU_${PU}.${JOBID}.root --nThreads ${NTHREADS}
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
from FWCore.ParameterSet.VarParsing import VarParsing

# In the line below 'analysis' is an instance of VarParsing object 
options = VarParsing ('analysis')

# Here we have defined our own two VarParsing options 
# add a list of strings for events to process
options.register ('nEvents',
                                 10,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.int,
                  "The number of events to simulate: 10")
options.register ('pileupFile',
                                 'file:/afs/cern.ch/work/g/gauzinge/public/minBias300k.root',
                                 VarParsing.multiplicity.list,
                                 VarParsing.varType.string,
                                 "File with Minimum Bias events to use as PU overlay")
options.register ('pileupAverage',
                                 10,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.float,
                  "The average PU number: 10")
options.register ('bunchSpace',
                                 25,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.int,
                  "The bunch spacing: 25 ns")
options.register ('minBunch',
                                 -12,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.int,
                  "The minimum bunch: -12 BX")
options.register ('maxBunch',
                                 3,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.int,
                  "The maximum bunch: 3 BX")
# options.register ('eventContent',
                                 # 'FEVTDEBUG',
                                 # VarParsing.multiplicity.singleton,
                                 # VarParsing.varType.string,
                  # "The Event content string: FEVTDEBUG")
options.register ('nThreads',
                                 1,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.int,
                  "The number of threads to use: 1")
options.register ('jobId',
                                 0,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.int,
                  "The job Id: 0")
options.register ('outputDirectory',
                  'file:/eos/user/g/gauzinge/PUdata',
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.string,
                  "The output directory")

options.parseArguments()
options.outputFile=options.outputDirectory+'/step3_pixel_PU_'+str(options.pileupAverage)+'.'+str(options.jobId)+'.root'
print("Output File: %s" % (options.outputFile))

process = cms.Process('FULLSIM',eras.Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
# process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D21Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D21_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
# process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
# process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
# process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# randomeze the seeds every time cmsRun is invoked
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()

# set up the event number
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.nEvents)
)

# Input source
process.source = cms.Source("EmptySource")

# process options
process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step1_step2_step3 nevts:'+str(options.nEvents)),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(options.outputFile),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
# include the filtering step
process.FEVTDEBUGoutput.outputCommands.append('drop  *')
process.FEVTDEBUGoutput.outputCommands.append('keep  *_g4SimHits__*')
# process.FEVTDEBUGoutput.outputCommands.append('keep  *_*_TrackerHitsPixelEndcapHighTof_*')
# process.FEVTDEBUGoutput.outputCommands.append('keep  *_*_TrackerHitsPixelEndcapLowTof_*')
# process.FEVTDEBUGoutput.outputCommands.append('keep  *_*_TrackerHitsPixelBarrelHighTof_*')
# process.FEVTDEBUGoutput.outputCommands.append('keep  *_*_TrackerHitsPixelBarrelLowTof_*')
process.FEVTDEBUGoutput.outputCommands.append('keep  *_simSiPixelDigis_Pixel_*')
process.FEVTDEBUGoutput.outputCommands.append('keep  *_siPixelClusters_*_*')

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# generator setup
process.generator = cms.EDProducer("FlatRandomEGunProducer",
    AddAntiParticle = cms.bool(False),
    PGunParameters = cms.PSet(
        MaxE = cms.double(10.01),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinE = cms.double(9.99),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359),
        PartID = cms.vint32(12)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single Nu E 10')
)

# mixing module
process.mix.input.nbPileupEvents.averageNumber = cms.double(options.pileupAverage)
process.mix.bunchspace = cms.int32(options.bunchSpace)
process.mix.minBunch = cms.int32(options.minBunch)
process.mix.maxBunch = cms.int32(options.maxBunch)
# process.mix.seed = cms.int32(@SEED@)
# process.mix.input.fileNames = cms.untracked.vstring([options.pileupFile])
process.mix.input.fileNames = cms.untracked.vstring(options.pileupFile)
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
# process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
# process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
# process.recosim_step = cms.Path(process.recosim)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.digi2raw_step,process.raw2digi_step,process.reconstruction_step,process.endjob_step,process.FEVTDEBUGoutput_step)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(options.nThreads)
process.options.numberOfStreams=cms.untracked.uint32(options.nThreads)

# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path).insert(0, process.generator)

#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)
# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

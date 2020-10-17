# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --mc --conditions auto:phase2_realistic --pileup_input file:myPUfile -n ${NEVENTS} --era Phase2 --eventcontent ${FEVTDEBUG} -s DIGI:pdigi_valid,L1,DIGI2RAW --datatier GEN-SIM-DIGI-RAW --pileup ${PUSTRING} --geometry Extended2023D21 --filein file:step1.root --fileout file:step2.root --nThreads ${NTHREADS}
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

options.parseArguments()

process = cms.Process('DIGI2RAW',eras.Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D21Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.nEvents)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:step1.root'),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_genParticles_*_*', 
        'drop *_genParticlesForJets_*_*', 
        'drop *_kt4GenJets_*_*', 
        'drop *_kt6GenJets_*_*', 
        'drop *_iterativeCone5GenJets_*_*', 
        'drop *_ak4GenJets_*_*', 
        'drop *_ak7GenJets_*_*', 
        'drop *_ak8GenJets_*_*', 
        'drop *_ak4GenJetsNoNu_*_*', 
        'drop *_ak8GenJetsNoNu_*_*', 
        'drop *_genCandidatesForMET_*_*', 
        'drop *_genParticlesForMETAllVisible_*_*', 
        'drop *_genMetCalo_*_*', 
        'drop *_genMetCaloAndNonPrompt_*_*', 
        'drop *_genMetTrue_*_*', 
        'drop *_genMetIC5GenJs_*_*'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:'+str(options.nEvents)),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.mix.input.nbPileupEvents.averageNumber = cms.double(options.pileupAverage)
process.mix.bunchspace = cms.int32(options.bunchSpace)
process.mix.minBunch = cms.int32(options.minBunch)
process.mix.maxBunch = cms.int32(options.maxBunch)
# process.mix.seed = cms.int32(@SEED@)
# process.mix.input.fileNames = cms.untracked.vstring([options.pileupFile])
process.mix.input.fileNames = cms.untracked.vstring(options.pileupFile)
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.endjob_step,process.FEVTDEBUGoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(options.nThreads)
process.options.numberOfStreams=cms.untracked.uint32(options.nThreads)

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

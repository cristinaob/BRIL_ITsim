###########################################
#
# BH_generator.py
#
# Test script for MIB generation (generate 1000 B1 events
# using either FLUKA or MARS inputs)
#
# To use is just do
#
# cmsRun BH_generator.py
#
# SV: 20/12/2010
#
##########################################

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

from Configuration.StandardSequences.Eras import eras

# In the line below 'analysis' is an instance of VarParsing object 
options = VarParsing ('analysis')

# Here we have defined our own two VarParsing options 
# add a list of strings for events to process
options.register ('nEvents',
                                 # 1000,
                                 -1,
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.int,
                  "The number of events to generate: 10")
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
                  'file:/afs/cern.ch/user/g/gauzinge/BIBSim/CMSSW_11_2_0_pre6/src/BRIL_ITsim/BIBGeneration/',
                  # 'file:/afs/cern.ch/work/g/gauzinge/public/',
                                 VarParsing.multiplicity.singleton,
                                 VarParsing.varType.string,
                  "The output directory")

options.parseArguments()
options.outputFile=options.outputDirectory+'/BeamHalo.'+str(options.jobId)+'.root'
print("Output File: %s" % (options.outputFile))

process = cms.Process('GEN', eras.Phase2) # Generation only

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
# process.load('Configuration.Geometry.GeometryExtended2023D21Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2023D21_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
# process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('Configuration.StandardSequences.VtxSmearedNoSmear_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('BRIL_ITsim.DataProductionTkOnly.cmsExtendedGeometry2026D999XML_cff')
print 'Running with special BRIL Tk Only Geometry'

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.168.2.1 $'),
    annotation = cms.untracked.string('Test script for MIB production'),
    name = cms.untracked.string('PyReleaseValidation'),
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.nEvents)
)

# Input source
process.source = cms.Source("EmptySource")


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string(options.outputFile),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN'),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Global tal clearly depends on the release you are working on
# To get the correct GT, look at this page:
#
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFrontierConditions?redirectedfrom=CMS.SWGuideFrontierConditions#Global_Tags_for_Monte_Carlo_Prod
#

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


# Here we choose the input, look into MIB_generator_cfi for more infos about the inputs
 
# process.load('BRIL_BIBGenerator.GeneratorInterface.BeamHaloGenerator.MIB_generator_cff')
process.load('GeneratorInterface.BeamHaloGenerator.MIB_generator_cff')

# process.Tracer          = cms.Service("Tracer")
process.generator       = process.FLUKA_generator.clone()  # FLUKA
# process.generator       = process.MARS_generator   # MARS
process.generator.InputFile = cms.string('/afs/cern.ch/work/g/gauzinge/public/BeamHalo/run0001_hilumi_ir5_exp_SCO001_fort.30')


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.endjob_step     = cms.Path(process.endOfProcess)
process.out_step        = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.endjob_step,process.out_step)

# special treatment in case of production filter sequence  
for path in process.paths:     
    getattr(process,path)._seq = process.generator*getattr(process,path)._seq


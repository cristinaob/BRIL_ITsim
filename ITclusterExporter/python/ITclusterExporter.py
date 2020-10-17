# imports
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


# create a new CMS process
process = cms.Process("ITclusterExporter")

# set up the options
options = VarParsing.VarParsing('analysis')
#set up the defaults
options.inputFiles = 'file:/eos/user/g/gauzinge/PUdata/step3_pixel_PU_175.0.root'
options.maxEvents = 1000 #all events

#get and parse command line arguments
options.parseArguments()

# load the standard
# process.load('Configuration.Geometry.GeometryExtended2023D21Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D63Reco_cff')
#custom BRIL configs like Geometry
# process.load('BRIL_ITsim.DataProductionTkOnly.cmsExtendedGeometry2026D999XML_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('ITclusterExporter')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

# the input file
process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring(options.inputFiles)
                            )

# the config of my analyzer
process.BRIL_IT_Analysis = cms.EDAnalyzer('ITclusterExporter',
                                         clusters=cms.InputTag("siPixelClusters"),
                                         disk=cms.untracked.uint32(1)
                                         )

# the TFIleService that produces the output root files
# process.TFileService = cms.Service("TFileService",
                                   # fileName=cms.string(options.outputFile)
                                   # )


process.p = cms.Path(process.BRIL_IT_Analysis)

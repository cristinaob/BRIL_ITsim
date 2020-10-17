# imports
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

# create a new CMS process
process = cms.Process("ITclusterAnalyzer")

# set up the options
options = VarParsing.VarParsing('analysis')
#set up the defaults
# options.inputFiles = 'file:/eos/user/g/gauzinge/PUdata/step3_pixel_PU_1.1.root'
options.inputFiles = 'file:/afs/cern.ch/user/g/gauzinge/BIBSim/CMSSW_11_2_0_pre6/src/BRIL_ITsim/DataProductionTkOnly/step3_pixel_PU_100.0.0TkOnly.root'
# options.inputFiles = 'file:/afs/cern.ch/work/c/cbarrera/private/BRIL/outputDir/step3_pixel_PU_20.0.0.root'
options.outputFile='summary.root'
options.maxEvents = -1 #all events

#get and parse command line arguments
options.parseArguments()

# load standard Geometry
#process.load('Configuration.Geometry.GeometryExtended2023D21Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D63Reco_cff')
#custom BRIL configs like Geometry
# process.load('BRIL_ITsim.DataProductionTkOnly.cmsExtendedGeometry2026D999XML_cff')
# process.load('Configuration.StandardSequences.MagneticField_cff')

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('ITclusterAnalyzer')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    limit=cms.untracked.int32(-1)
)
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False)
                                    # ,SkipEvent = cms.untracked.vstring('ProductNotFound')
                                    )

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

# the input file
process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring(options.inputFiles)
                            )

process.content = cms.EDAnalyzer("EventContentAnalyzer")
# the config of my analyzer
process.BRIL_IT_Analysis = cms.EDAnalyzer('ITclusterAnalyzer',
                                         clusters=cms.InputTag("siPixelClustersPreSplitting"),
                                         simlinks=cms.InputTag("simSiPixelDigis", "Pixel", "FULLSIM"),
                                         digis=cms.InputTag("simSiPixelDigis", "Pixel", "FULLSIM"),
                                         # simlinks=cms.InputTag("Pixel"),
                                         # simtracks=cms.InputTag("g4SimHits"),
                                         maxBin=cms.untracked.uint32(5000),
                                         docoincidence=cms.untracked.bool(True),
                                         storeClusterTree=cms.untracked.bool(False),
                                         dx_cut=cms.double(0.1),
                                         dy_cut=cms.double(0.1),
                                         dz_cut=cms.double(0.9)
                                         )

# the TFIleService that produces the output root files
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string(options.outputFile)
                                   )


# process.p = cms.Path( ... process.content * ...  )
process.p = cms.Path(process.BRIL_IT_Analysis)

from __future__ import print_function
import FWCore.ParameterSet.Config as cms

def TkOnlyDigi(process):
    print("!!! Special version of the digitization for tracker only !!!")
    if hasattr(process,'digitisationTkOnly_step'):
        process=customise_DigiTkOnly(process)

    return process

def customise_DigiTkOnly(process):
    process.load('Configuration.StandardSequences.Digi_cff')
    process.doAllDigi = cms.Sequence()
    process.load('SimGeneral.MixingModule.mixObjects_cfi')
    process.digitisationTkOnly_step.remove(process.mix.mixObjects.mixCH)
    del process.simCastorDigis
    #ecal
    del process.simEcalUnsuppressedDigis
    del process.simEcalTriggerPrimitiveDigis
    del process.simEcalDigis
    del process.simEcalPreshowerDigis
    #hcal
    del process.simHcalUnsuppressedDigis
    del process.simHcalTriggerPrimitiveDigis
    del process.simHcalDigis
    del process.simHcalTTPDigis
    #hgc
    del process.simHGCalUnsuppressedDigis
    #muon
    del process.simMuonCSCDigis
    del process.simMuonDTDigis
    del process.simMuonRPCDigis
    del process.simMuonME0Digis
    del process.simMuonME0PseudoDigis
    del process.simMuonME0PseudoReDigis
    del process.simMuonGEMDigis
    process.mix.digitizers = cms.PSet(process.theDigitizersValid)
    del process.mix.digitizers.ecal
    del process.mix.digitizers.hcal
    del process.mix.digitizers.calotruth
    # del process.mix.digitizers.hcal
    # del process.mix.digitizers.castor
    del process.mix.digitizers.ecalTime
    del process.mix.digitizers.hgceeDigitizer
    del process.mix.digitizers.hgchebackDigitizer
    del process.mix.digitizers.hgchefrontDigitizer
    del process.mix.digitizers.fastTimingLayer
    # print(process.mix.digitizers)
    process.digitisationTkOnly_step.remove(process.mix.digitizers.pixel)
    process.load('SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi')
    process.mix.digitizers.pixel=process.phase2TrackerDigitizer
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    #Check if mergedtruth is in the sequence first, could be taken out depending on cmsDriver options
    if hasattr(process.mix.digitizers,"mergedtruth") :
        process.mix.digitizers.mergedtruth.simHitCollections.muon = cms.VInputTag( )
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"))
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"))
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"))
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"))
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECLowTof"))
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECHighTof"))
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"))
        # process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"))

    # keep new digis
    alist=['FEVTDEBUG','FEVTDEBUGHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep Phase2TrackerDigiedmDetSetVector_*_*_*')
    return process


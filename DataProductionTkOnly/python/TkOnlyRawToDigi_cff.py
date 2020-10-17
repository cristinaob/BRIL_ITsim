import FWCore.ParameterSet.Config as cms

# This object is used to selectively make changes for different running
# scenarios. In this case it makes changes for Run 2.

#Simplified BRIL version, skimmed by G. Auzinger, 23/09/2020

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *

from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *

# from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *

# from EventFilter.EcalRawToDigi.ecalDigis_cff import *

# import EventFilter.ESRawToDigi.esRawToDigi_cfi
# ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()

# import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
# hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()

# import EventFilter.CSCRawToDigi.cscUnpacker_cfi
# muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()

# import EventFilter.DTRawToDigi.dtunpacker_cfi
# muonDTDigis = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone()

# import EventFilter.RPCRawToDigi.RPCRawToDigi_cfi 
# muonRPCDigis = EventFilter.RPCRawToDigi.RPCRawToDigi_cfi.muonRPCDigis.clone()

# import EventFilter.GEMRawToDigi.muonGEMDigis_cfi
# muonGEMDigis = EventFilter.GEMRawToDigi.muonGEMDigis_cfi.muonGEMDigis.clone()

# from EventFilter.CastorRawToDigi.CastorRawToDigi_cff import *
# castorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone( FEDs = cms.untracked.vint32(690,691,692, 693,722) )

# from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import *

from EventFilter.Utilities.tcdsRawToDigi_cfi import *
tcdsDigis = EventFilter.Utilities.tcdsRawToDigi_cfi.tcdsRawToDigi.clone()

from EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi import *
onlineMetaDataDigis = EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi.onlineMetaDataRawToDigi.clone()

# from L1Trigger.Configuration.L1TRawToDigi_cff import *

# from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import *

# RawToDigiTask = cms.Task(L1TRawToDigiTask,
                         # siPixelDigis,
                         # siStripDigis,
                         # ecalDigisTask,
                         # ecalPreshowerDigis,
                         # hcalDigis,
                         # muonCSCDigis,
                         # muonDTDigis,
                         # muonRPCDigis,
                         # castorDigis,
                         # scalersRawToDigi,
                         # tcdsDigis,
                         # onlineMetaDataDigis,
                         # )
RawToDigiTask = cms.Task(#L1TRawToDigiTask,
                         siPixelDigis,
                         # siStripDigis,
                         # ecalDigisTask,
                         # ecalPreshowerDigis,
                         # hcalDigis,
                         # muonCSCDigis,
                         # muonDTDigis,
                         # muonRPCDigis,
                         # castorDigis,
                         # scalersRawToDigi,
                         tcdsDigis,
                         onlineMetaDataDigis,
                         )
RawToDigi = cms.Sequence(RawToDigiTask)

# RawToDigiTask_noTk = RawToDigiTask.copyAndExclude([siPixelDigis, siStripDigis])
# RawToDigi_noTk = cms.Sequence(RawToDigiTask_noTk)

RawToDigiTask_pixelOnly = cms.Task(siPixelDigis)
RawToDigi_pixelOnly = cms.Sequence(RawToDigiTask_pixelOnly)

# RawToDigiTask_ecalOnly = cms.Task(ecalDigisTask, ecalPreshowerDigis, scalersRawToDigi)
# RawToDigi_ecalOnly = cms.Sequence(RawToDigiTask_ecalOnly)

# RawToDigiTask_hcalOnly = cms.Task(hcalDigis)
# RawToDigi_hcalOnly = cms.Sequence(RawToDigiTask_hcalOnly)

# scalersRawToDigi.scalersInputTag = 'rawDataCollector'
siPixelDigis.InputLabel = 'rawDataCollector'
# ecalDigis.InputLabel = 'rawDataCollector'
# ecalPreshowerDigis.sourceTag = 'rawDataCollector'
# hcalDigis.InputLabel = 'rawDataCollector'
# muonCSCDigis.InputObjects = 'rawDataCollector'
# muonDTDigis.inputLabel = 'rawDataCollector'
# muonRPCDigis.InputLabel = 'rawDataCollector'
# castorDigis.InputLabel = 'rawDataCollector'

# from Configuration.Eras.Modifier_run3_common_cff import run3_common
# run3_common.toReplaceWith(RawToDigiTask, RawToDigiTask.copyAndExclude([castorDigis]))

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# Remove siPixelDigis until we have phase1 pixel digis
phase2_tracker.toReplaceWith(RawToDigiTask, RawToDigiTask.copyAndExclude([siPixelDigis])) # FIXME


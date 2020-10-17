import FWCore.ParameterSet.Config as cms

# This object is used to make changes for different running scenarios. In
# this case for Run 2
#Simplified BRIL version, skimmed by G. Auzinger, 23/09/2020

from EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
DigiToRawTask = cms.Task(siPixelRawData, SiStripDigiToRaw)#, ecalPacker, esDigiToRaw, hcalRawDataTask, cscpacker, dtpacker, rpcpacker, castorRawData, rawDataCollector)
DigiToRaw = cms.Sequence(DigiToRawTask)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(DigiToRawTask, DigiToRawTask.copyAndExclude([siPixelRawData])) # FIXME


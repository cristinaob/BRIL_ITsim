// -*- C++ -*-
//
// Package:    BRIL_ITsim/ITclusterExporter
// Class:      ITclusterExporter
//
/**\class ITclusterExporter ITclusterExporter.cc BRIL_ITsim/ITclusterExporter/plugins/ITclusterExporter.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Georg Auzinger
//         Created:  Thu, 17 Jan 2019 13:12:52 GMT
//
//

// system include files
// system include files
#include <fstream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
//#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

class ITclusterExporter : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
    explicit ITclusterExporter(const edm::ParameterSet&);
    ~ITclusterExporter();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    // ----------member data ---------------------------
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> m_tokenClusters;

    // the pointers to geometry, topology and clusters
    // these are members so all functions can access them without passing as argument
    const TrackerTopology* tTopo = NULL;
    const TrackerGeometry* tkGeom = NULL;
    const edmNew::DetSetVector<SiPixelCluster>* clusters = NULL;
    //from config file
    uint32_t m_disk;
    uint32_t m_module;
    //event counter
    uint32_t m_nevents;
    //fstreams
    std::ofstream ring1;
    std::ofstream ring2;
    std::ofstream ring3;
    std::ofstream ring4;
    std::ofstream ring5;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ITclusterExporter::ITclusterExporter(const edm::ParameterSet& iConfig)
    : m_tokenClusters(consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("clusters")))
    , m_disk(iConfig.getUntrackedParameter<uint32_t>("disk"))
{
    //now do what ever initialization is needed
    m_nevents = 0;
    m_module = 1;
}

ITclusterExporter::~ITclusterExporter()
{

    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void ITclusterExporter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    //get the clusters
    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> tclusters;
    iEvent.getByToken(m_tokenClusters, tclusters);

    // Get the geometry
    edm::ESHandle<TrackerGeometry> tgeomHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get("idealForDigi", tgeomHandle);

    // Get the topology
    edm::ESHandle<TrackerTopology> tTopoHandle;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);

    //get the pointers to geometry, topology and clusters
    tTopo = tTopoHandle.product();
    const TrackerGeometry* tkGeom = &(*tgeomHandle);
    tkGeom = tgeomHandle.product();
    clusters = tclusters.product();

    //loop the modules in the cluster collection
    for (typename edmNew::DetSetVector<SiPixelCluster>::const_iterator DSVit = clusters->begin(); DSVit != clusters->end(); DSVit++) {
        //get the detid
        unsigned int rawid(DSVit->detId());
        DetId detId(rawid);

        //figure out the module type using the detID
        TrackerGeometry::ModuleType mType = tkGeom->getDetectorType(detId);
        if (mType != TrackerGeometry::ModuleType::Ph2PXF && detId.subdetId() != PixelSubdetector::PixelEndcap)
            continue;
        //std::cout << "DetID " << std::hex << "0x" << rawid << std::dec << " " << detId.det() << " " << detId.subdetId() << " " << ((rawid >> 23) & 0x3) << " " << ((rawid >> 18) & 0xF) << " " << ((rawid >> 12) & 0x3F) << " " << ((rawid >> 2) & 0xFF) << std::endl;

        //find out which layer, side and ring
        unsigned int side = (tTopo->pxfSide(detId));  // values are 1 and 2 for -+Z
        unsigned int layer = (tTopo->pxfDisk(detId)); //values are 1 to 12 for disks TFPX1 to TFPX 8  and TEPX1 to TEPX 4
        unsigned int ring = (tTopo->pxfBlade(detId));
        unsigned int module = (tTopo->pxfModule(detId));

        //now make sure we are only looking at TEPX
        if (side != 1)
            continue;
        if (layer < 9)
            continue;
        else if (layer - 8 != m_disk)
            continue;

        if (module != m_module)
            continue;

        //now I am sure that only modules m_module in disk m_disk pass
        std::cout << "Disk: " << layer - 8 << " Ring " << ring << " Module " << module << std::endl;

        // Get the geomdet
        //const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
        //if (!geomDetUnit)
        //continue;

        // the number of clusters for this module with DetID detId
        unsigned int nClu = DSVit->size();

        if (ring == 1)
            ring1 << "Event " << nClu << std::endl;
        else if (ring == 2)
            ring2 << "Event " << nClu << std::endl;
        else if (ring == 3)
            ring3 << "Event " << nClu << std::endl;
        else if (ring == 4)
            ring4 << "Event " << nClu << std::endl;
        else if (ring == 5)
            ring5 << "Event " << nClu << std::endl;

        unsigned int cluCounter = 0;

        //now loop the clusters for each detector
        for (edmNew::DetSet<SiPixelCluster>::const_iterator cluit = DSVit->begin(); cluit != DSVit->end(); cluit++) {
            // determine the position
            //MeasurementPoint mpClu(cluit->x(), cluit->y());
            //Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);
            //Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

            //now get the list of hits
            //care for  size(in pixels),
            //list of
            //x coordinate in pixels, y coordinate, adc, cluster number
            int size = cluit->size();
            for (int i = 0; i < size; i++) {
                SiPixelCluster::Pixel pix = cluit->pixel(i);
                //               f << p.x << ' ' << p.y << ' ' << p.adc << ' ' << i << '\n';
                if (ring == 1)
                    ring1 << pix.x << " " << pix.y << " " << pix.adc << " " << cluCounter << std::endl;
                else if (ring == 2)
                    ring2 << pix.x << " " << pix.y << " " << pix.adc << " " << cluCounter << std::endl;
                else if (ring == 3)
                    ring3 << pix.x << " " << pix.y << " " << pix.adc << " " << cluCounter << std::endl;
                else if (ring == 4)
                    ring4 << pix.x << " " << pix.y << " " << pix.adc << " " << cluCounter << std::endl;
                else if (ring == 5)
                    ring5 << pix.x << " " << pix.y << " " << pix.adc << " " << cluCounter << std::endl;
            }
            cluCounter++;

        } //end of cluster loop
    }     //end of module loop

    m_nevents++;
}

// ------------ method called once each job just before starting event loop  ------------
void ITclusterExporter::beginJob()
{
    //open the filestreams
    std::stringstream filename;
    filename << "Clusters_Disk" << m_disk << "_Ring1_Module" << m_module << ".txt";
    ring1.open(filename.str().c_str(), std::ofstream::out);
    filename.str("");
    filename << "Clusters_Disk" << m_disk << "_Ring2_Module" << m_module << ".txt";
    ring2.open(filename.str().c_str(), std::ofstream::out);
    filename.str("");
    filename << "Clusters_Disk" << m_disk << "_Ring3_Module" << m_module << ".txt";
    ring3.open(filename.str().c_str(), std::ofstream::out);
    filename.str("");
    filename << "Clusters_Disk" << m_disk << "_Ring4_Module" << m_module << ".txt";
    ring4.open(filename.str().c_str(), std::ofstream::out);
    filename.str("");
    filename << "Clusters_Disk" << m_disk << "_Ring5_Module" << m_module << ".txt";
    ring5.open(filename.str().c_str(), std::ofstream::out);
    filename.str("");
}

// ------------ method called once each job just after ending the event loop  ------------
void ITclusterExporter::endJob()
{
    ring1.close();
    ring2.close();
    ring3.close();
    ring4.close();
    ring5.close();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ITclusterExporter::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);

    //Specify that only 'tracks' is allowed
    //To use, remove the default given above and uncomment below
    //ParameterSetDescription desc;
    //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
    //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ITclusterExporter);

//SiPixelCluster clusters[N];

//and perhaps if/when they will exists
////SiNoiseHits noise[M];
//
//int32_t ntotalhits = 0;
//
//for ( int i = 0; i != N; ++i  )
//  ntotalhits += clusters[i].size();
//
//  //ntotalhits += M;
//
//  ofstream f("file.csv");
//
//  f << ntotalhits << '\n';
//
//  for ( int i = 0; i != N; ++i  )
//    for ( int j = 0; j != clusters[i].size(); ++j  )
//       {
//           SiPixelCluster::Pixel p = clusters[i].pixel( j  );
//               f << p.x << ' ' << p.y << ' ' << p.adc << ' ' << i << '\n';
//                  }
//
//                  //for ( int k = 0; i != M; ++k  )
//                  //  f << noise[k].x << ' ' << noise[k].y << ' ' << noise[k].adc << ' ' << -1 << '\n';
//       }

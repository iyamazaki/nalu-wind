// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "HypreLinearSystem.h"
#include "HypreDirectSolver.h"
#include "Realm.h"
#include "EquationSystem.h"
#include "LinearSolver.h"
#include "PeriodicManager.h"
#include "NaluEnv.h"
#include "NonConformalManager.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"

// NGP Algorithms
#include "ngp_utils/NgpLoopUtils.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "krylov.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE.h"
#include "HYPRE_config.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>

namespace sierra {
namespace nalu {

HypreLinearSystem::HypreLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver)
  : LinearSystem(realm, numDof, eqSys, linearSolver),
    name_(eqSys->name_), numAssembles_(0), numUnfilledRows_(0),
    rowFilled_(0),
    rowStatus_(0),
    idBuffer_(0)
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  rows_.clear();
  cols_.clear();
  vals_.clear();
  rhsRows_.clear();
  rhsVals_.clear();
  count_.clear();
  partitionCount_.clear();
  printf("Done %s %s %d : name=%s\n\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

HypreLinearSystem::~HypreLinearSystem()
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);
    HYPRE_IJVectorDestroy(rhs_);
    HYPRE_IJVectorDestroy(sln_);
    systemInitialized_ = false;
  }
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreLinearSystem::beginLinearSystemConstruction()
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  if (inConstruction_) return;
  inConstruction_ = true;

#ifndef HYPRE_BIGINT
  // Make sure that HYPRE is compiled with 64-bit integer support when running
  // O(~1B) linear systems.
  uint64_t totalRows = (static_cast<uint64_t>(realm_.hypreNumNodes_) *
                        static_cast<uint64_t>(numDof_));
  uint64_t maxHypreSize = static_cast<uint64_t>(std::numeric_limits<HypreIntType>::max());

  if (totalRows > maxHypreSize)
    throw std::runtime_error(
      "The linear system size is greater than what HYPRE is compiled for. "
      "Please recompile with bigint support and link to Nalu");
#endif

  const int rank = realm_.bulk_data().parallel_rank();

  if (rank == 0) {
    iLower_ = realm_.hypreILower_;
  } else {
    iLower_ = realm_.hypreILower_ * numDof_ ;
  }

  iUpper_ = realm_.hypreIUpper_  * numDof_ - 1;
  // For now set column indices the same as row indices
  jLower_ = iLower_;
  jUpper_ = iUpper_;

  // The total number of rows handled by this MPI rank for Hypre
  numRows_ = (iUpper_ - iLower_ + 1);
  // Total number of global rows in the system
  maxRowID_ = realm_.hypreNumNodes_ * numDof_ - 1;

#if 1
  if (numDof_ > 0)
    std::cerr << rank << "\t" << numDof_ << "\t"
              << realm_.hypreILower_ << "\t" << realm_.hypreIUpper_ << "\t"
                << iLower_ << "\t" << iUpper_ << "\t"
                << numRows_ << "\t" << maxRowID_ << std::endl;
#endif
  // Allocate memory for the arrays used to track row types and row filled status.
  rowFilled_.resize(numRows_);
  rowStatus_.resize(numRows_);
  skippedRows_.clear();
  // All nodes start out as NORMAL; "build*NodeGraph" methods might alter the
  // row status to modify behavior of sumInto method.
  for (HypreIntType i=0; i < numRows_; i++)
    rowStatus_[i] = RT_NORMAL;

  auto& bulk = realm_.bulk_data();
  std::vector<const stk::mesh::FieldBase*> fVec{realm_.hypreGlobalId_};

  stk::mesh::copy_owned_to_shared(bulk, fVec);
  stk::mesh::communicate_field_data(bulk.aura_ghosting(), fVec);

  if (realm_.oversetManager_ != nullptr &&
      realm_.oversetManager_->oversetGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.oversetManager_->oversetGhosting_, fVec);

  if (realm_.nonConformalManager_ != nullptr &&
      realm_.nonConformalManager_->nonConformalGhosting_ != nullptr)
    stk::mesh::communicate_field_data(
      *realm_.nonConformalManager_->nonConformalGhosting_, fVec);

  if (realm_.periodicManager_ != nullptr &&
      realm_.periodicManager_->periodicGhosting_ != nullptr) {
    realm_.periodicManager_->parallel_communicate_field(realm_.hypreGlobalId_);
    realm_.periodicManager_->periodic_parallel_communicate_field(
      realm_.hypreGlobalId_);
  }
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreLinearSystem::buildNodeGraph(
  const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
  beginLinearSystemConstruction();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_owned );
  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      count = count<0 ? numDof_*numDof_ : count;
    }
  }

  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}


void
HypreLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
  beginLinearSystemConstruction();


  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets = realm_.get_buckets(realm_.meta_data().side_rank(), s_owned);
  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      const unsigned numNodes = b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes*numNodes*numDof_*numDof_) : count;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
  beginLinearSystemConstruction();

  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::EDGE_RANK, s_owned);
  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      const unsigned numNodes = b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes*numNodes*numDof_*numDof_) : count;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
  beginLinearSystemConstruction();

  stk::mesh::MetaData & metaData = realm_.meta_data();
  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
                                      & stk::mesh::selectUnion(parts)
                                      & !(realm_.get_inactive_selector());
  stk::mesh::BucketVector const& buckets = realm_.get_buckets(stk::topology::ELEM_RANK, s_owned);
  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  for(size_t ib=0; ib<buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      /* fill temporaries */
      index++;
      const unsigned numNodes = b.num_nodes(k);
      count = count<0 ? (HypreIntType)(numNodes*numNodes*numDof_*numDof_) : count;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
  beginLinearSystemConstruction();

  stk::mesh::BulkData & bulkData = realm_.bulk_data();
  stk::mesh::MetaData & metaData = realm_.meta_data();

  const stk::mesh::Selector s_owned = metaData.locally_owned_part()
    & stk::mesh::selectUnion(parts)
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( metaData.side_rank(), s_owned );

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=-1;
  for(size_t ib=0; ib<face_buckets.size(); ++ib) {
    const stk::mesh::Bucket & b = *face_buckets[ib];
    for ( stk::mesh::Bucket::size_type k = 0 ; k < b.size() ; ++k ) {
      const stk::mesh::Entity face = b[k];

      // extract the connected element to this exposed face; should be single in size!
      const stk::mesh::Entity* face_elem_rels = bulkData.begin_elements(face);
      ThrowAssert( bulkData.num_elements(face) == 1 );

      // get connected element and nodal relations
      stk::mesh::Entity element = face_elem_rels[0];
      const stk::mesh::Entity* elem_nodes = bulkData.begin_nodes(element);

      // figure out the global dof ids for each dof on each node
      const size_t numNodes = bulkData.num_nodes(element);
      const unsigned nn = numNodes*numNodes*numDof_*numDof_;
      count = count<nn ? (HypreIntType)(nn) : count;
      index++;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  beginLinearSystemConstruction();
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  beginLinearSystemConstruction();
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreLinearSystem::buildOversetNodeGraph(
  const stk::mesh::PartVector&)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
  // rows during assembly process
  for(auto* oinfo: realm_.oversetManager_->oversetInfoVec_) {
    auto node = oinfo->orphanNode_;
    HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
    skippedRows_.insert(hid * numDof_);
  } 
  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  // Grab nodes regardless of whether they are owned or shared
  const stk::mesh::Selector sel = stk::mesh::selectUnion(parts);
  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=1;
  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
      skippedRows_.insert(hid * numDof_);
      index++;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
}

void
HypreLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=1;
  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    skippedRows_.insert(hid * numDof_);
    index++;
  }

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());

  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
}

void 
HypreLinearSystem::buildDirichletNodeGraph(const ngp::Mesh::ConnectedNodes nodeList) {
  printf("%s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;
  printf("%d\n",nodeList.size());
  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=1;
  for (unsigned i=0; i<nodeList.size();++i) {
    HypreIntType hid = get_entity_hypre_id(nodeList[i]);
    printf("%d %d %d\n",i,nodeList.size(),(int)hid);
    skippedRows_.insert(hid * numDof_);
    index++;
  }

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());

  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
}

void
HypreLinearSystem::finalizeLinearSystem()
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  ThrowRequire(inConstruction_);
  inConstruction_ = false;

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  finalizeSolver();

  /* create these mappings */
  fill_entity_to_row_mapping();

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;

  // At this stage the LHS and RHS data structures are ready for
  // sumInto/assembly.
  systemInitialized_ = true;
  printf("Done %s %s %d : name=%s\n\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreLinearSystem::finalizeSolver()
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_);
  HYPRE_IJVectorSetObjectType(rhs_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_);
  HYPRE_IJVectorSetObjectType(sln_, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}


void HypreLinearSystem::fill_entity_to_row_mapping()
{
  const stk::mesh::BulkData& bulk = realm_.bulk_data();
  stk::mesh::Selector selector = bulk.mesh_meta_data().universal_part() & !(realm_.get_inactive_selector());
  entityToLID_ = EntityToHypreIntTypeView("entityToRowLID",bulk.get_size_of_entity_index_space());

  const stk::mesh::BucketVector& nodeBuckets = realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for(const stk::mesh::Bucket* bptr : nodeBuckets) {
    const stk::mesh::Bucket& b = *bptr;
    for(size_t i=0; i<b.size(); ++i) {
      stk::mesh::Entity node = b[i];
      const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
      const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluId);
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);
      entityToLID_[node.local_offset()] = hid;
    }
  }
}

void
HypreLinearSystem::loadComplete()
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  // All algorithms have called sumInto and populated LHS/RHS. Now we are ready
  // to finalize the matrix at the HYPRE end. However, before we do that we need
  // to process unfilled rows and process them appropriately. Any row acted on
  // by sumInto method will have toggled the rowFilled_ array to RS_FILLED
  // status. Before finalizing assembly, we process rows that still have an
  // RS_UNFILLED status and set their diagonal entries to 1.0 (dummy row)
  //
  // TODO: Alternate design to eliminate dummy rows. This will require
  // load-balancing on HYPRE end.
  printf("%s %s %d : name=%s\n\tnumAssembles_=%d, size of lists before unfilled = rows=%lu, cols=%lu, vals=%lu\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numAssembles_,rows_.size(),cols_.size(),vals_.size());

  HypreIntType hnrows = 1;
  HypreIntType hncols = 1;
  double getval;
  double setval = 1.0;
  numUnfilledRows_=0;

  for (HypreIntType i=0; i < numRows_; i++) {
    if (rowFilled_[i] == RS_FILLED) continue;
    else numUnfilledRows_++;
  }

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d, numUnfilledRows=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size(),(int)numUnfilledRows_);

  numUnfilledRows_=0;

  for (HypreIntType i=0; i < numRows_; i++) {
    if (rowFilled_[i] == RS_FILLED) continue;
    HypreIntType lid = iLower_ + i;
    HYPRE_IJMatrixGetValues(mat_, hnrows, &hncols, &lid, &lid, &getval);
    if (std::fabs(getval) < 1.0e-12) {
      HYPRE_IJMatrixSetValues(mat_, hnrows, &hncols, &lid, &lid, &setval);
      rows_.push_back(lid);
      cols_.push_back(lid);
      vals_.push_back(setval);
      for (unsigned j=0;j<rhsRows_.size();++j) {
	rhsRows_[j].push_back(lid);
	rhsVals_[j].push_back(0.0);
      }
      numUnfilledRows_++;
    }
  }

  if (name_=="ContinuityEQS" || name_=="WallDistEQS" || name_=="TurbKineticEnergyEQS" || name_=="MomentumEQS") {
    newHostCoeffApplier->finishAssembly();
  }

  printf("Done %s %s %d : name=%s, numDof=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d, numUnfilledRows=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size(),(int)numUnfilledRows_);

  numAssembles_++;

  loadCompleteSolver();

  printf("\t%s %s %d : name=%s\n\tnumAssembles_=%d, size of lists (Old way) = rows=%lu, cols=%lu, vals=%lu\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),numAssembles_,rows_.size(),cols_.size(),vals_.size());

  if (name_=="ContinuityEQS" || name_=="WallDistEQS" || name_=="TurbKineticEnergyEQS" || name_=="MomentumEQS") {
    newHostCoeffApplier->dumpData(numAssembles_);
  }
  
  char fname[50];
  sprintf(fname,"%s_rowIndices%d.bin",name_.c_str(),numAssembles_);
  std::ofstream rfile(fname, std::ios::out | std::ios::binary);
  rfile.write((char*)&rows_[0], rows_.size() * sizeof(HypreIntType));
  rfile.close();
    
  sprintf(fname,"%s_colIndices%d.bin",name_.c_str(),numAssembles_);
  std::ofstream cfile(fname, std::ios::out | std::ios::binary);
  cfile.write((char*)&cols_[0], cols_.size() * sizeof(HypreIntType));
  cfile.close();
    
  sprintf(fname,"%s_values%d.bin",name_.c_str(),numAssembles_);
  std::ofstream vfile(fname, std::ios::out | std::ios::binary);
  vfile.write((char*)&vals_[0], vals_.size() * sizeof(double));
  vfile.close();

  for (unsigned i=0;i<rhsRows_.size();++i) {
    sprintf(fname,"%s_rhsRowIndices%d_%u.bin",name_.c_str(),numAssembles_,i);
    std::ofstream rrfile(fname, std::ios::out | std::ios::binary);
    rrfile.write((char*)&(rhsRows_[i][0]), rhsRows_[i].size() * sizeof(HypreIntType));
    rrfile.close();
    
    sprintf(fname,"%s_rhsValues%d_%u.bin",name_.c_str(),numAssembles_,i);
    std::ofstream rvfile(fname, std::ios::out | std::ios::binary);
    rvfile.write((char*)&(rhsVals_[i][0]), rhsVals_[i].size() * sizeof(double));
    rvfile.close();
  }
   
  std::vector<HypreIntType> metaData(0);
  metaData.push_back((HypreIntType)iLower_);
  metaData.push_back((HypreIntType)iUpper_);
  metaData.push_back((HypreIntType)jLower_);
  metaData.push_back((HypreIntType)jUpper_);
  metaData.push_back((HypreIntType)vals_.size());
  metaData.push_back((HypreIntType)rhsVals_[0].size());
  sprintf(fname,"%s_metaData%d.bin",name_.c_str(),numAssembles_);
  std::ofstream mdfile(fname, std::ios::out | std::ios::binary);
  long pos = mdfile.tellp();
  int size = sizeof(HypreIntType);
  mdfile.write((char *)&size, 4);
  mdfile.seekp(pos+4);
  mdfile.write((char*)&metaData[0], metaData.size() * sizeof(HypreIntType));
  mdfile.close();
  
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreLinearSystem::loadCompleteSolver()
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  HYPRE_IJVectorAssemble(rhs_);
  HYPRE_IJVectorGetObject(rhs_, (void**)&(solver->parRhs_));

  HYPRE_IJVectorAssemble(sln_);
  HYPRE_IJVectorGetObject(sln_, (void**)&(solver->parSln_));

  solver->comm_ = realm_.bulk_data().parallel();
    
  hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag((HYPRE_ParCSRMatrix) (solver->parMat_));
  HYPRE_Int * hr = (HYPRE_Int *) hypre_CSRMatrixI(diag);
  HYPRE_Int * hc = (HYPRE_Int *) hypre_CSRMatrixJ(diag);
  double * hd = (double *) hypre_CSRMatrixData(diag);
  HYPRE_Int num_rows = diag->num_rows;
  HYPRE_Int num_nonzeros = diag->num_nonzeros;
    
  char fname[50];
  sprintf(fname,"%s_HypreRows%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hrfile(fname, std::ios::out | std::ios::binary);
  std::vector<HypreIntType> tmp(num_rows+1);
  for (int i=0; i<num_rows+1; ++i) { tmp[i] = (HypreIntType)hr[i]; }
  hrfile.write((char*)&tmp[0], (num_rows+1) * sizeof(HypreIntType));
  hrfile.close();
  
  sprintf(fname,"%s_HypreCols%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hcfile(fname, std::ios::out | std::ios::binary);
  tmp.resize(num_nonzeros);
  for (int i=0; i<num_nonzeros; ++i) { tmp[i] = (HypreIntType)hc[i]; }
  hcfile.write((char*)&tmp[0], num_nonzeros * sizeof(HypreIntType));
  hcfile.close();
  
  sprintf(fname,"%s_HypreData%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hdfile(fname, std::ios::out | std::ios::binary);
  hdfile.write((char*)&hd[0], num_nonzeros * sizeof(double));
  hdfile.close();
  
  double * local_data = hypre_VectorData(hypre_ParVectorLocalVector(solver->parRhs_));
  sprintf(fname,"%s_HypreRHSData%d_0.bin",name_.c_str(),numAssembles_);
  std::ofstream hrhsfile(fname, std::ios::out | std::ios::binary);
  hrhsfile.write((char*)local_data, num_rows * sizeof(double));
  hrhsfile.close();

  std::vector<HypreIntType> hmetaData(0);
  hmetaData.push_back((HypreIntType)num_rows);
  hmetaData.push_back((HypreIntType)num_nonzeros);
  sprintf(fname,"%s_HypreMetaData%d.bin",name_.c_str(),numAssembles_);
  std::ofstream hmdfile(fname, std::ios::out | std::ios::binary);
  long pos = hmdfile.tellp();
  int size = sizeof(HypreIntType);
  hmdfile.write((char *)&size, 4);
  hmdfile.seekp(pos+4);
  hmdfile.write((char*)&hmetaData[0], hmetaData.size() * sizeof(HypreIntType));
  hmdfile.close();

  // Set flag to indicate zeroSystem that the matrix must be reinitialized
  // during the next invocation.
  matrixAssembled_ = true;
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreLinearSystem::zeroSystem()
{
  printf("\n\nZero System\n%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(linearSolver_);

  rows_.clear();
  cols_.clear();
  vals_.clear();
  rhsRows_.clear();
  rhsVals_.clear();
  rhsRows_.resize(1);
  rhsVals_.resize(1);
  rhsRows_[0].clear();
  rhsVals_[0].clear();

  // It is unsafe to call IJMatrixInitialize multiple times without intervening
  // call to IJMatrixAssemble. This occurs during the first outer iteration (of
  // first timestep in static application and every timestep in moving mesh
  // applications) when the data structures have been created but never used and
  // zeroSystem is called for a reset. Include a check to ensure we only
  // initialize if it was previously assembled.
  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    HYPRE_IJVectorInitialize(rhs_);
    HYPRE_IJVectorInitialize(sln_);

    // Set flag to false until next invocation of IJMatrixAssemble in loadComplete
    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parRhs_, 0.0);
  HYPRE_ParVectorSetConstantValues(solver->parSln_, 0.0);

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

sierra::nalu::CoeffApplier* HypreLinearSystem::get_new_coeff_applier()
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  if (!newHostCoeffApplier) {

    unsigned numPartitions = partitionCount_.size();
    printf("%s %s %d : name=%s numPartitions=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)numPartitions);
    Kokkos::View<HypreIntType *> partitionCountView = Kokkos::View<HypreIntType *>("partitionCount",numPartitions);
    Kokkos::View<HypreIntType *> countView = Kokkos::View<HypreIntType *>("count",numPartitions);

    for (unsigned i=0; i<numPartitions; ++i) {
      partitionCountView[i] = partitionCount_[i];
      countView[i] = count_[i];
    }

    /* skipped rows data structure */
    Kokkos::UnorderedMap<HypreIntType,HypreIntType> skippedRowsMap(skippedRows_.size());
    for (auto t : skippedRows_) {
      skippedRowsMap.insert(t,t);
    }
    printf("%s %s %d : name=%s skippedRowsMap size=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)skippedRowsMap.size());

    // Total number of global rows in the system
    HypreIntType maxRowID = realm_.hypreNumNodes_ * numDof_ - 1;
    
    newHostCoeffApplier.reset(new HypreLinSysCoeffApplier(name_, numDof_, numPartitions, maxRowID,
							  iLower_, iUpper_, jLower_, jUpper_,
                                                          partitionCountView, countView, 
							  entityToLID_, skippedRowsMap));
    newDeviceCoeffApplier = newHostCoeffApplier->device_pointer();

    /* clear this data so that the next time a coeffApplier is built, these get rebuilt from scratch */
    partitionCount_.clear();
    count_.clear();
  }
  /* reset the internal counters */
  newHostCoeffApplier->resetInternalData();
  
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  return newDeviceCoeffApplier;
}

/********************************************************************************************************/
/*                     Beginning of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/
HypreLinearSystem::HypreLinSysCoeffApplier::HypreLinSysCoeffApplier(std::string name, unsigned numDof, 
								    unsigned numPartitions, HypreIntType maxRowID,
								    HypreIntType iLower, HypreIntType iUpper,
								    HypreIntType jLower, HypreIntType jUpper,
								    Kokkos::View<HypreIntType *> partitionCount,
								    Kokkos::View<HypreIntType *> mat_count,
								    EntityToHypreIntTypeView entityToLID,
								    Kokkos::UnorderedMap<HypreIntType,HypreIntType> skippedRowsMap)
  : name_(name), numDof_(numDof), numPartitions_(numPartitions), maxRowID_(maxRowID),
    iLower_(iLower), iUpper_(iUpper), jLower_(jLower), jUpper_(jUpper),
    partitionCount_(partitionCount), mat_count_(mat_count),
    entityToLID_(entityToLID), skippedRowsMap_(skippedRowsMap),
    devicePointer_(nullptr)
{
  printf("%s %s %d : name=%s : numDof_=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_);
  
  // The total number of rows handled by this MPI rank for Hypre
  numRows_ = (iUpper_ - iLower_ + 1);
  
  /* set key internal data */
  partitionIndex_=-1;
  numMatPtsToAssembleTotal_=0;
  numRhsPtsToAssembleTotal_=0;

  /* meta data that allows one to write to different parts of the lists from different
     Assemble*SolverAlgorithm calls. These are effectively the partitions */
  rhs_count_ = Kokkos::View<HypreIntType *>("rows",numPartitions_);
  mat_partitionStart_ = Kokkos::View<HypreIntType *>("rows",numPartitions_);
  rhs_partitionStart_ = Kokkos::View<HypreIntType *>("rows",numPartitions_);
  for (unsigned i=0; i<numPartitions_; ++i) {
    rhs_count_[i] = sqrt(mat_count_[i]);
    numMatPtsToAssembleTotal_ += partitionCount_[i]*mat_count_[i];
    numRhsPtsToAssembleTotal_ += partitionCount_[i]*rhs_count_[i];
    mat_partitionStart_[i] = numMatPtsToAssembleTotal_ - partitionCount_[i]*mat_count_[i];
    rhs_partitionStart_[i] = numRhsPtsToAssembleTotal_ - partitionCount_[i]*rhs_count_[i];
    printf("%d : total=%d, partitionCount=%d, count=%d, partitionStart=%d\n",
	   i,(int)numMatPtsToAssembleTotal_,(int)partitionCount_[i],(int)mat_count_[i],(int)mat_partitionStart_[i]);
  }      

  /* storage for the matrix lists */
  rows_ = Kokkos::View<HypreIntType *>("rows",numMatPtsToAssembleTotal_ + numRows_);
  cols_ = Kokkos::View<HypreIntType *>("cols",numMatPtsToAssembleTotal_ + numRows_);
  vals_ = Kokkos::View<double *>("vals",numMatPtsToAssembleTotal_ + numRows_);
  Kokkos::parallel_for("initLists", numMatPtsToAssembleTotal_, KOKKOS_LAMBDA (const int& i) {
      /* initialize to the dummy value -1 so that row and cols entries in the list that aren't "filled in"
	 are easily ignored during the full assembly process */
      rows_[i] = -1;
      cols_[i] = -1;
      vals_[i] = 0.;
    });

  /* storage for the rhs lists */
  rhsRows_ = Kokkos::View<HypreIntType *>("rhsRows",numRhsPtsToAssembleTotal_ + numRows_);
  rhsVals_ = Kokkos::View<double *>("rhsVvals",numRhsPtsToAssembleTotal_ + numRows_);
  Kokkos::parallel_for("initLists", numRhsPtsToAssembleTotal_, KOKKOS_LAMBDA (const int& i) {
      /* initialize to the dummy value -1 so that row and cols entries in the list that aren't "filled in"
	 are easily ignored during the full assembly process */
      rhsRows_[i] = -1;
      rhsVals_[i] = 0.;
    });

  /* initialize the row filled status vector */
  rowFilled_ = Kokkos::View<RowFillStatus*>("rowFilled",numRows_);  
  Kokkos::parallel_for("initRowFilled", numRows_, KOKKOS_LAMBDA (const int& i) {
      rowFilled_[i] = RS_UNFILLED;
    });

  /* create the mirror ... need for the bc hack */
  rowFilledHost_ = Kokkos::create_mirror_view(rowFilled_);

  /* define the atomic for the matrix */
  mat_atomic_counter_ = Kokkos::View<HypreIntType>("mat_counter");
  mat_atomic_counter_() = 0;

  /* define the atomic for the rhs */
  rhs_atomic_counter_ = Kokkos::View<HypreIntType>("rhs_counter");
  rhs_atomic_counter_() = 0;

  /* check skipped rows */
  checkSkippedRows_ = skippedRowsMap_.size()>0 ? true : false;
  
  printf("Done %s %s %d : name=%s : numDof_=%d, numMatPtsToAssemble=%d, checkSkippedRows=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),numDof_,(int)numMatPtsToAssembleTotal_,(int)checkSkippedRows_);
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::operator()(
  unsigned numEntities,
  const ngp::Mesh::ConnectedNodes& entities,
  const SharedMemView<int*, DeviceShmem>& localIds,
  const SharedMemView<int*, DeviceShmem>& sortPermutation,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const char* /*trace_tag*/)
{
  //int myLocalIndex = Kokkos::atomic_fetch_add(&atomic_counter_(), 1);
  //HypreIntType matIndex = mat_partitionStart_[partitionIndex_] + myLocalIndex*mat_count_[partitionIndex_];
  //HypreIntType rhsIndex = rhs_partitionStart_[partitionIndex_] + myLocalIndex*rhs_count_[partitionIndex_];
  int matIndex = Kokkos::atomic_fetch_add(&mat_atomic_counter_(), mat_count_[partitionIndex_]);
  int rhsIndex = Kokkos::atomic_fetch_add(&rhs_atomic_counter_(), rhs_count_[partitionIndex_]);

  HypreIntType numRows = numEntities * numDof_;
  for(unsigned i = 0; i < numEntities; i++) {
    HypreIntType hid = entityToLID_[entities[i].local_offset()];
    for(unsigned d=0; d < numDof_; ++d) {
      unsigned lid = i*numDof_ + d;
      localIds[lid] = hid*numDof_ + d;
    }
  }

  for (unsigned i=0; i < numEntities; i++) {
    int ix = i * numDof_;
    HypreIntType hid = localIds[ix];
    if (checkSkippedRows_) {
      if (skippedRowsMap_.exists(hid)) continue;
    }
    for (unsigned d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = localIds[ir];
      const double* cur_lhs = &lhs(ir, 0);

      /* fill the matrix values */
      for (int k=0; k<numRows; ++k) {
	rows_[matIndex+i*numRows+k] = lid;
	cols_[matIndex+i*numRows+k] = localIds[k];
	vals_[matIndex+i*numRows+k] = cur_lhs[k];
      }
      /* fill the right hand side values */
      rhsRows_[rhsIndex+i] = lid;
      rhsVals_[rhsIndex+i] = rhs[ir];

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
}

void HypreLinearSystem::HypreLinSysCoeffApplier::dumpData(const int di) {
  int matCount = (int) mat_atomic_counter_();
  int rhsCount = (int) rhs_atomic_counter_();
  printf("matCount=%d, rhsCount=%d\n",matCount,rhsCount);

  char fname[50];
  sprintf(fname,"%s_CoeffApplier_rowIndices%d.bin",name_.c_str(),di);
  std::ofstream rfile(fname, std::ios::out | std::ios::binary);
  rfile.write((char*)&rows_[0], matCount * sizeof(HypreIntType));
  rfile.close();
    
  sprintf(fname,"%s_CoeffApplier_colIndices%d.bin",name_.c_str(),di);
  std::ofstream cfile(fname, std::ios::out | std::ios::binary);
  cfile.write((char*)&cols_[0], matCount * sizeof(HypreIntType));
  cfile.close();
    
  sprintf(fname,"%s_CoeffApplier_values%d.bin",name_.c_str(),di);
  std::ofstream vfile(fname, std::ios::out | std::ios::binary);
  vfile.write((char*)&vals_[0], matCount * sizeof(double));
  vfile.close();

  sprintf(fname,"%s_CoeffApplier_rhsRowIndices%d_0.bin",name_.c_str(),di);
  std::ofstream rrfile(fname, std::ios::out | std::ios::binary);
  rrfile.write((char*)&rhsRows_[0], rhsCount * sizeof(HypreIntType));
  rrfile.close();

  sprintf(fname,"%s_CoeffApplier_rhsValues%d_0.bin",name_.c_str(),di);
  std::ofstream vvfile(fname, std::ios::out | std::ios::binary);
  vvfile.write((char*)&rhsVals_[0], rhsCount * sizeof(double));
  vvfile.close();

  std::vector<HypreIntType> metaData(0);
  metaData.push_back((HypreIntType)iLower_);
  metaData.push_back((HypreIntType)iUpper_);
  metaData.push_back((HypreIntType)jLower_);
  metaData.push_back((HypreIntType)jUpper_);
  metaData.push_back((HypreIntType)matCount);
  metaData.push_back((HypreIntType)rhsCount);
  sprintf(fname,"%s_CoeffApplier_metaData%d.bin",name_.c_str(),di);
  std::ofstream mdfile(fname, std::ios::out | std::ios::binary);
  long pos = mdfile.tellp();
  int size = sizeof(HypreIntType);
  mdfile.write((char *)&size, 4);
  mdfile.seekp(pos+4);
  mdfile.write((char*)&metaData[0], metaData.size() * sizeof(HypreIntType));
  mdfile.close();
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
							      stk::mesh::FieldBase * solutionField,
							      stk::mesh::FieldBase * bcValuesField,
							      const stk::mesh::PartVector& parts) {

  printf("%s %s %d : mat_counter=%d\n",__FILE__,__FUNCTION__,__LINE__,(int)mat_atomic_counter_());

  resetInternalData();

#if 1

  /************************************************************/
  /* this is a hack to get dirichlet bcs working consistently */

  /* Step 1: copy the rowFilled_ to its host mirror */
  Kokkos::deep_copy(rowFilledHost_, rowFilled_);

  /* Step 2: execute the old CPU code */

  auto& meta = realm.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm.get_inactive_selector()));

  const auto& bkts = realm.get_buckets(
    stk::topology::NODE_RANK, sel);

  double diag_value = 1.0;
  std::vector<HypreIntType> tRows(0);
  std::vector<HypreIntType> tCols(0);
  std::vector<double> tVals(0);
  std::vector<HypreIntType> trhsRows(0);
  std::vector<double> trhsVals(0);

  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm.hypreGlobalId_, node);

      for (size_t d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];

	/* fill the mirrored version */
        rowFilledHost_[lid - iLower_] = RS_FILLED;

	/* fill these temp values */
	tRows.push_back(lid);
	tCols.push_back(lid);
	tVals.push_back(diag_value);
	trhsRows.push_back(lid);
	trhsVals.push_back(bcval);
      }
    }
  }

  /* Step 3 : allocate space in which to push the temporaries */
  Kokkos::View<HypreIntType*> r("r",tRows.size());
  Kokkos::View<HypreIntType*>::HostMirror rh  = Kokkos::create_mirror_view(r);
  Kokkos::View<HypreIntType*> c("c",tCols.size());
  Kokkos::View<HypreIntType*>::HostMirror ch  = Kokkos::create_mirror_view(c);
  Kokkos::View<double*> v("v",tVals.size());
  Kokkos::View<double*>::HostMirror vh  = Kokkos::create_mirror_view(v);
  Kokkos::View<HypreIntType*> rr("rr",trhsRows.size());
  Kokkos::View<HypreIntType*>::HostMirror rrh  = Kokkos::create_mirror_view(rr);
  Kokkos::View<double*> rv("rv",trhsVals.size());
  Kokkos::View<double*>::HostMirror rvh  = Kokkos::create_mirror_view(rv);

  /* Step 4 : next copy the std::vectors into the host mirrors */
  for (unsigned int i=0; i<tRows.size(); ++i) {
    rh[i] = tRows[i];
    ch[i] = tCols[i];
    vh[i] = tVals[i];
    rrh[i] = trhsRows[i];
    rvh[i] = trhsVals[i];
  }

  /* Step 5 : deep copy this to device */
  Kokkos::deep_copy(rowFilled_,rowFilledHost_);
  Kokkos::deep_copy(r,rh);
  Kokkos::deep_copy(c,ch);
  Kokkos::deep_copy(v,vh);
  Kokkos::deep_copy(rr,rrh);
  Kokkos::deep_copy(rv,rvh);

  /* Step 6 : append this to the existing data structure */
  Kokkos::parallel_for("bcHack", tRows.size(), KOKKOS_LAMBDA (const int& i) {
      int matIndex = Kokkos::atomic_fetch_add(&mat_atomic_counter_(), 1);
      int rhsIndex = Kokkos::atomic_fetch_add(&rhs_atomic_counter_(), 1);
      rows_[matIndex]=r[i];
      cols_[matIndex]=c[i];
      vals_[matIndex]=v[i];
      rhsRows_[rhsIndex] = rr[i];
      rhsVals_[rhsIndex] = rv[i];
    });

#else

  stk::mesh::MetaData & metaData = realm.meta_data();

  const stk::mesh::Selector selector = (
    metaData.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm.get_inactive_selector()));

  using MeshIndex = nalu_ngp::NGPMeshTraits<ngp::Mesh>::MeshIndex;

  ngp::Mesh ngpMesh = realm.ngp_mesh();
  NGPDoubleFieldType ngpSolutionField = realm.ngp_field_manager().get_field<double>(solutionField->mesh_meta_data_ordinal());
  NGPDoubleFieldType ngpBCValuesField = realm.ngp_field_manager().get_field<double>(bcValuesField->mesh_meta_data_ordinal());

  ngpSolutionField.sync_to_device();
  ngpBCValuesField.sync_to_device();

  nalu_ngp::run_entity_algorithm(
    "HypreLinSysCoeffApplier::applyDirichletBCs", ngpMesh, stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(const MeshIndex& meshIdx)
    {
      stk::mesh::Entity entity = (*meshIdx.bucket)[meshIdx.bucketOrd];
      HypreIntType hid = entityToLID_[entity.local_offset()];

      int matIndex = Kokkos::atomic_fetch_add(&mat_atomic_counter_(), numDof_);
      int rhsIndex = Kokkos::atomic_fetch_add(&rhs_atomic_counter_(), numDof_);      
      double diag_value = 1.0;
      printf("hid=%d\n",(int)hid);
      
      for (size_t d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        const double bc_residual = ngpBCValuesField.get(meshIdx, d) - ngpSolutionField.get(meshIdx, d);
	rows_[matIndex+d] = lid;
	cols_[matIndex+d] = lid;
	vals_[matIndex+d] = diag_value;
	rhsRows_[rhsIndex+d] = lid;
	rhsVals_[rhsIndex+d] = bc_residual;
	rowFilled_[lid - iLower_] = RS_FILLED;
	
      }
    }
  );

#endif
  printf("Done %s %s %d : mat_counter=%d\n",__FILE__,__FUNCTION__,__LINE__,(int)mat_atomic_counter_());
}

void
HypreLinearSystem::HypreLinSysCoeffApplier::finishAssembly() {

  printf("%s %s %d : name=%s : mat_atomic_counter_()=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)mat_atomic_counter_());

  /* initialize the row filled status vector */
  Kokkos::parallel_for("unfilledRows", numRows_, KOKKOS_LAMBDA (const int& i) {
      if (rowFilled_[i]==RS_UNFILLED) {
	int matIndex = Kokkos::atomic_fetch_add(&mat_atomic_counter_(), 1);
	int rhsIndex = Kokkos::atomic_fetch_add(&rhs_atomic_counter_(), 1);
	HypreIntType lid = iLower_ + i;
	rows_[matIndex] = lid;
	cols_[matIndex] = lid;
	vals_[matIndex] = 1.0;
	rhsRows_[rhsIndex] = lid;
	rhsVals_[rhsIndex] = 0.0;
      }
    }); 

  printf("Done %s %s %d : name=%s : mat_atomic_counter_()=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)mat_atomic_counter_());
}


void
HypreLinearSystem::HypreLinSysCoeffApplier::resetInternalData() {
  if (numPartitions_>0) {
    printf("%s %s %d : name=%s : partitionIndex_=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),partitionIndex_);
    partitionIndex_++;
    partitionIndex_= (partitionIndex_%numPartitions_);
    checkSkippedRows_=true;
    if (partitionIndex_==0) {
      mat_atomic_counter_() = 0;
      rhs_atomic_counter_() = 0;
      Kokkos::parallel_for("resetRowFilled", numRows_, KOKKOS_LAMBDA (const int& i) {
	  rowFilled_[i] = RS_UNFILLED;
	});
    }
    printf("Done %s %s %d : name=%s : partitionIndex_=%d, mat_atomic_counter_()=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),partitionIndex_,(int)mat_atomic_counter_());
  }
}

void HypreLinearSystem::HypreLinSysCoeffApplier::free_device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (this != devicePointer_) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
#endif
}

sierra::nalu::CoeffApplier* HypreLinearSystem::HypreLinSysCoeffApplier::device_pointer()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (devicePointer_ != nullptr) {
    sierra::nalu::kokkos_free_on_device(devicePointer_);
    devicePointer_ = nullptr;
  }
  devicePointer_ = sierra::nalu::create_device_expression(*this);
#else
  devicePointer_ = this;
#endif
  return devicePointer_;
}

/********************************************************************************************************/
/*                           End of HypreLinSysCoeffApplier implementations                             */
/********************************************************************************************************/


void
HypreLinearSystem::sumInto(
  unsigned numEntities,
  const ngp::Mesh::ConnectedNodes& entities,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  const size_t n_obj = numEntities;
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      const double* cur_lhs = &lhs(ir, 0);
      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], cur_lhs);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

      for (int k=0; k<numRows; ++k) {
	rows_.push_back(lid);
	cols_.push_back(idBuffer_[k]);
	vals_.push_back(cur_lhs[k]);
      }
      rhsRows_[0].push_back(lid);
      rhsVals_[0].push_back(rhs[ir]);

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
#endif
}


void
HypreLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj * numDof_;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssert(numRows == static_cast<HypreIntType>(rhs.size()));
  ThrowAssert(numRows*numRows == static_cast<HypreIntType>(lhs.size()));

  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    HypreIntType hid = get_entity_hypre_id(entities[in]);
    HypreIntType localOffset = hid * numDof_;
    for (size_t d=0; d < numDof_; d++) {
      size_t lid = in * numDof_ + d;
      idBuffer_[lid] = localOffset + d;
    }
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * numDof_;
    HypreIntType hid = idBuffer_[ix];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    for (size_t d=0; d < numDof_; d++) {
      int ir = ix + d;
      HypreIntType lid = idBuffer_[ir];

      for (int c=0; c < numRows; c++)
        scratchVals[c] = lhs[ir * numRows + c];

      HYPRE_IJMatrixAddToValues(mat_, 1, &numRows, &lid,
                                &idBuffer_[0], &scratchVals[0]);
      HYPRE_IJVectorAddToValues(rhs_, 1, &lid, &rhs[ir]);

      for (int k=0; k<numRows; ++k) {
	rows_.push_back(lid);
	cols_.push_back(idBuffer_[k]);
	vals_.push_back(scratchVals[k]);
      }
      rhsRows_[0].push_back(lid);
      rhsVals_[0].push_back(rhs[ir]);

      if ((lid >= iLower_) && (lid <= iUpper_))
        rowFilled_[lid - iLower_] = RS_FILLED;
    }
  }
}

void
HypreLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  printf("%s %s %d : name=%s, list length=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)rows_.size());

  double adbc_time = -NaluEnv::self().nalu_time();

  auto& meta = realm_.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm_.get_inactive_selector()));

  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

  std::vector<int> crap3(0);
  HypreIntType ncols = 1;
  double diag_value = 1.0;
  for (auto b: bkts) {
    const double* solution = (double*)stk::mesh::field_data(
      *solutionField, *b);
    const double* bcValues = (double*)stk::mesh::field_data(
      *bcValuesField, *b);

    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);

      for (size_t d=0; d<numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        double bcval = bcValues[in*numDof_ + d] - solution[in*numDof_ + d];

        HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &lid, &lid, &diag_value);
        HYPRE_IJVectorSetValues(rhs_, 1, &lid, &bcval);
        rowFilled_[lid - iLower_] = RS_FILLED;

	rows_.push_back(lid);
	cols_.push_back(lid);
	vals_.push_back(diag_value);
	rhsRows_[0].push_back(lid);
	rhsVals_[0].push_back(bcval);
      }
    }
  }

  adbc_time += NaluEnv::self().nalu_time();

  if (name_=="ContinuityEQS" || name_=="WallDistEQS" || name_=="TurbKineticEnergyEQS" || name_=="MomentumEQS") {
    newHostCoeffApplier->applyDirichletBCs(realm_, solutionField, bcValuesField, parts);
  }

  printf("Done %s %s %d : name=%s, list length=%d\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str(),(int)rows_.size());
}

HypreIntType
HypreLinearSystem::get_entity_hypre_id(const stk::mesh::Entity& node)
{
  auto& bulk = realm_.bulk_data();
  const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
  const auto mnode = bulk.get_entity(stk::topology::NODE_RANK, naluId);
#ifndef NDEBUG
  if (!bulk.is_valid(node))
    throw std::runtime_error("BAD STK NODE");
#endif
  HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);

#ifndef NDEBUG
  HypreIntType chk = ((hid+1) * numDof_ - 1);
  if ((hid < 0) || (chk > maxRowID_)) {
    std::cerr << bulk.parallel_rank() << "\t"
              << hid << "\t" << iLower_ << "\t" << iUpper_ << std::endl;
    throw std::runtime_error("BAD STK to hypre conversion");
  }
#endif

  return hid;
}

HypreIntType
HypreLinearSystem::get_entity_hypre_id(stk::mesh::EntityRank rank,
				       const stk::mesh::Entity& node)
{
  auto& bulk = realm_.bulk_data();
  const auto naluId = *stk::mesh::field_data(*realm_.naluGlobalId_, node);
  const auto mnode = bulk.get_entity(rank, naluId);
  //#ifndef NDEBUG
  if (!bulk.is_valid(node))
    throw std::runtime_error("BAD STK NODE");
  //#endif
  HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, mnode);

#ifndef NDEBUG
  HypreIntType chk = ((hid+1) * numDof_ - 1);
  if ((hid < 0) || (chk > maxRowID_)) {
    std::cerr << bulk.parallel_rank() << "\t"
              << hid << "\t" << iLower_ << "\t" << iUpper_ << std::endl;
    throw std::runtime_error("BAD STK to hypre conversion");
  }
#endif

  return hid;
}

int
HypreLinearSystem::solve(stk::mesh::FieldBase* linearSolutionField)
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());

  HypreDirectSolver* solver = reinterpret_cast<HypreDirectSolver*>(
    linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    const std::string rhsFile = eqSysName_ + ".IJV." + writeCounter + ".rhs";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());
    HYPRE_IJVectorPrint(rhs_, rhsFile.c_str());
  }

  int iters = 0;
  double finalResidNorm = 0.0;

  // Call solve
  int status = 0;

  status = solver->solve(iters, finalResidNorm, realm_.isFinalOuterIter_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string slnFile = eqSysName_ + ".IJV." + writeCounter + ".sln";
    HYPRE_IJVectorPrint(sln_, slnFile.c_str());
    ++eqSys_->linsysWriteCounter_;
  }

  double norm2 = copy_hypre_to_stk(linearSolutionField);
  sync_field(linearSolutionField);

  linearSolveIterations_ = iters;
  // Hypre provides relative residuals not the final residual, so multiply by
  // the non-linear residual to obtain a final residual that is comparable to
  // what is reported by TpetraLinearSystem. Note that this assumes the initial
  // solution vector is set to 0 at the start of linear iterations.
  linearResidual_ = finalResidNorm * norm2;
  nonLinearResidual_ = realm_.l2Scaling_ * norm2;

  if (eqSys_->firstTimeStepSolve_)
    firstNonLinearResidual_ = nonLinearResidual_;

  scaledNonLinearResidual_ =
    nonLinearResidual_ /
    std::max(std::numeric_limits<double>::epsilon(), firstNonLinearResidual_);

  if (provideOutput_) {
    const int nameOffset = eqSysName_.length() + 8;
    NaluEnv::self().naluOutputP0()
      << std::setw(nameOffset) << std::right << eqSysName_
      << std::setw(32 - nameOffset) << std::right << iters << std::setw(18)
      << std::right << linearResidual_ << std::setw(15) << std::right
      << nonLinearResidual_ << std::setw(14) << std::right
      << scaledNonLinearResidual_ << std::endl;
  }

  eqSys_->firstTimeStepSolve_ = false;
  printf("Done %s %s %d : name=%s\n\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  return status;
}

double
HypreLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  double lclnorm2 = 0.0;
  double rhsVal = 0.0;
  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (size_t d=0; d < numDof_; d++) {
        HypreIntType lid = hid * numDof_ + d;
        int sid = in * numDof_ + d;
        HYPRE_IJVectorGetValues(sln_, 1, &lid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_, 1, &lid, &rhsVal);

        lclnorm2 += rhsVal * rhsVal;
      }
    }
  }

  double gblnorm2 = 0.0;
  stk::all_reduce_sum(bulk.parallel(), &lclnorm2, &gblnorm2, 1);

  return std::sqrt(gblnorm2);
}

}  // nalu
}  // sierra

// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "HypreUVWLinearSystem.h"
#include "HypreUVWSolver.h"
#include "NaluEnv.h"
#include "Realm.h"
#include "EquationSystem.h"

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

#include <limits>
#include <vector>
#include <string>
#include <cmath>

namespace sierra {
namespace nalu {

HypreUVWLinearSystem::HypreUVWLinearSystem(
  Realm& realm,
  const unsigned numDof,
  EquationSystem* eqSys,
  LinearSolver* linearSolver
) : HypreLinearSystem(realm, 1, eqSys, linearSolver),
    rhs_(numDof, nullptr),
    sln_(numDof, nullptr),
    nDim_(numDof)
{}

HypreUVWLinearSystem::~HypreUVWLinearSystem()
{
  if (systemInitialized_) {
    HYPRE_IJMatrixDestroy(mat_);

    for (int i=0; i < nDim_; i++) {
      HYPRE_IJVectorDestroy(rhs_[i]);
      HYPRE_IJVectorDestroy(sln_[i]);
    }
  }
  systemInitialized_ = false;
}

void
HypreUVWLinearSystem::finalizeSolver()
{

  MPI_Comm comm = realm_.bulk_data().parallel();
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixCreate(comm, iLower_, iUpper_, jLower_, jUpper_, &mat_);
  HYPRE_IJMatrixSetObjectType(mat_, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (int i=0; i < nDim_; i++) {
    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &rhs_[i]);
    HYPRE_IJVectorSetObjectType(rhs_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorCreate(comm, iLower_, iUpper_, &sln_[i]);
    HYPRE_IJVectorSetObjectType(sln_[i], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }
}

void
HypreUVWLinearSystem::loadCompleteSolver()
{
  // Now perform HYPRE assembly so that the data structures are ready to be used
  // by the solvers/preconditioners.
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  HYPRE_IJMatrixAssemble(mat_);
  HYPRE_IJMatrixGetObject(mat_, (void**)&(solver->parMat_));

  for (int i=0; i < nDim_; i++) {
    HYPRE_IJVectorAssemble(rhs_[i]);
    HYPRE_IJVectorGetObject(rhs_[i], (void**)&(solver->parRhsU_[i]));

    HYPRE_IJVectorAssemble(sln_[i]);
    HYPRE_IJVectorGetObject(sln_[i], (void**)&(solver->parSlnU_[i]));
  }

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

  for (int i=0;i<nDim_; ++i) {
    double * local_data = hypre_VectorData(hypre_ParVectorLocalVector(solver->parRhsU_[i]));
    sprintf(fname,"%s_HypreRHSData%d_%d.bin",name_.c_str(),numAssembles_,i);
    std::ofstream hrhsfile(fname, std::ios::out | std::ios::binary);
    hrhsfile.write((char*)local_data, num_rows * sizeof(double));
    hrhsfile.close();
  }    
  
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

  solver->comm_ = realm_.bulk_data().parallel();

  matrixAssembled_ = true;
}

void
HypreUVWLinearSystem::zeroSystem()
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  rows_.clear();
  cols_.clear();
  vals_.clear();
  rhsRows_.clear();
  rhsVals_.clear();
  rhsRows_.resize(nDim_);
  rhsVals_.resize(nDim_);
  for (int i=0; i < nDim_; i++) {
    rhsRows_[i].resize(0);
    rhsVals_[i].resize(0);
  }

  if (matrixAssembled_) {
    HYPRE_IJMatrixInitialize(mat_);
    for (int i=0; i < nDim_; i++) {
      HYPRE_IJVectorInitialize(rhs_[i]);
      HYPRE_IJVectorInitialize(sln_[i]);
    }

    matrixAssembled_ = false;
  }

  HYPRE_IJMatrixSetConstantValues(mat_, 0.0);
  for (int i=0; i < nDim_; i++) {
    HYPRE_ParVectorSetConstantValues((solver->parRhsU_[i]), 0.0);
    HYPRE_ParVectorSetConstantValues((solver->parSlnU_[i]), 0.0);
  }

  // Prepare for matrix assembly and set all entry flags to "unfilled"
  for (HypreIntType i=0; i < numRows_; i++)
    rowFilled_[i] = RS_UNFILLED;

  // Set flag to indicate whether rows must be skipped during normal sumInto
  // process. For this to be activated, the linear system must have Dirichlet or
  // overset rows and they must be present on this processor
  if (hasSkippedRows_ && !skippedRows_.empty())
    checkSkippedRows_ = true;
}

void
HypreUVWLinearSystem::sumInto(
  unsigned numEntities,
  const ngp::Mesh::ConnectedNodes& entities,
  const SharedMemView<const double*, DeviceShmem>& rhs,
  const SharedMemView<const double**, DeviceShmem>& lhs,
  const SharedMemView<int*, DeviceShmem>&,
  const SharedMemView<int*, DeviceShmem>&,
  const char*  /* trace_tag */)
{
#ifndef KOKKOS_ENABLE_CUDA
  HypreIntType numRows = numEntities;
  const HypreIntType bufSize = idBuffer_.size();

  ThrowAssertMsg(lhs.span_is_contiguous(), "LHS assumed contiguous");
  ThrowAssertMsg(rhs.span_is_contiguous(), "RHS assumed contiguous");
  if (bufSize < numRows) {
    idBuffer_.resize(numRows);
    scratchRowVals_.resize(numRows);
  }

  for (size_t in=0; in < numEntities; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < numEntities; in++) {
    int ix = in * nDim_;
    HypreIntType hid = idBuffer_[in];

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    int offset = 0;
    for (int c=0; c < numRows; c++) {
      scratchRowVals_[c] = lhs(ix, offset);
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchRowVals_[0]);

    for (int k=0; k<numRows; ++k) {
      rows_.push_back(hid);
      cols_.push_back(idBuffer_[k]);
      vals_.push_back(scratchRowVals_[k]);
    }

    for (int d=0; d < nDim_; d++) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
      rhsRows_[d].push_back(hid);
      rhsVals_[d].push_back(rhs[ir]);
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
#endif
}

void
HypreUVWLinearSystem::sumInto(
  const std::vector<stk::mesh::Entity>& entities,
  std::vector<int>&  /* scratchIds */,
  std::vector<double>& scratchVals,
  const std::vector<double>& rhs,
  const std::vector<double>& lhs,
  const char*  /* trace_tag */)
{
  const size_t n_obj = entities.size();
  HypreIntType numRows = n_obj;
  const HypreIntType bufSize = idBuffer_.size();

#ifndef NDEBUG
  size_t vecSize = numRows * nDim_;
  ThrowAssert(vecSize == rhs.size());
  ThrowAssert(vecSize*vecSize == lhs.size());
#endif
  if (bufSize < numRows) idBuffer_.resize(numRows);

  for (size_t in=0; in < n_obj; in++) {
    idBuffer_[in] = get_entity_hypre_id(entities[in]);
  }

  for (size_t in=0; in < n_obj; in++) {
    int ix = in * nDim_;
    HypreIntType hid = get_entity_hypre_id(entities[in]);

    if (checkSkippedRows_) {
      auto it = skippedRows_.find(hid);
      if (it != skippedRows_.end()) continue;
    }

    int offset = 0;
    int ic = ix * numRows * nDim_;
    for (int c=0; c < numRows; c++) {
      scratchVals[c] = lhs[ic + offset];
      offset += nDim_;
    }
    HYPRE_IJMatrixAddToValues(
      mat_, 1, &numRows, &hid, &idBuffer_[0], &scratchVals[0]);

    for (int k=0; k<numRows; ++k) {
      rows_.push_back(hid);
      cols_.push_back(idBuffer_[k]);
      vals_.push_back(scratchRowVals_[k]);
    }

    for (int d = 0; d < nDim_; d++) {
      int ir = ix + d;
      HYPRE_IJVectorAddToValues(rhs_[d], 1, &hid, &rhs[ir]);
      rhsRows_[d].push_back(hid);
      rhsVals_[d].push_back(rhs[ir]);
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
}

void
HypreUVWLinearSystem::applyDirichletBCs(
  stk::mesh::FieldBase* solutionField,
  stk::mesh::FieldBase* bcValuesField,
  const stk::mesh::PartVector& parts,
  const unsigned,
  const unsigned)
{
  auto& meta = realm_.meta_data();

  const stk::mesh::Selector sel = (
    meta.locally_owned_part() &
    stk::mesh::selectUnion(parts) &
    stk::mesh::selectField(*solutionField) &
    !(realm_.get_inactive_selector()));

  const auto& bkts = realm_.get_buckets(
    stk::topology::NODE_RANK, sel);

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

      HYPRE_IJMatrixSetValues(mat_, 1, &ncols, &hid, &hid, &diag_value);
      rows_.push_back(hid);
      cols_.push_back(hid);
      vals_.push_back(diag_value);
      for (int d=0; d<nDim_; d++) {
        double bcval = bcValues[in*nDim_ + d] - solution[in*nDim_ + d];

        HYPRE_IJVectorSetValues(rhs_[d], 1, &hid, &bcval);
	rhsRows_[d].push_back(hid);
	rhsVals_[d].push_back(bcval);
      }
      rowFilled_[hid - iLower_] = RS_FILLED;
    }
  }
}

int
HypreUVWLinearSystem::solve(stk::mesh::FieldBase* slnField)
{
  HypreUVWSolver* solver = reinterpret_cast<HypreUVWSolver*>(linearSolver_);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
    const std::string matFile = eqSysName_ + ".IJM." + writeCounter + ".mat";
    HYPRE_IJMatrixPrint(mat_, matFile.c_str());

    for (int d=0; d < nDim_; d++) {
      const std::string rhsFile =
        eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".rhs";
      HYPRE_IJVectorPrint(rhs_[d], rhsFile.c_str());
    }
  }

  int status = 0;
  std::vector<int> iters(nDim_, 0);
  std::vector<double> finalNorm(nDim_, 1.0);
  std::vector<double> rhsNorm(nDim_, std::numeric_limits<double>::max());

  for (int d=0; d < nDim_; d++) {
    status = solver->solve(d, iters[d], finalNorm[d], realm_.isFinalOuterIter_);
  }
  copy_hypre_to_stk(slnField, rhsNorm);
  sync_field(slnField);

  if (solver->getConfig()->getWriteMatrixFiles()) {
    for (int d=0; d < nDim_; d++) {
      std::string writeCounter = std::to_string(eqSys_->linsysWriteCounter_);
      const std::string slnFile = eqSysName_ + std::to_string(d) + ".IJV." + writeCounter + ".sln";
      HYPRE_IJVectorPrint(sln_[d], slnFile.c_str());
      ++eqSys_->linsysWriteCounter_;
    }
  }

  {
    linearSolveIterations_ = 0;
    linearResidual_ = 0.0;
    nonLinearResidual_ = 0.0;
    double linres, nonlinres, scaledres, tmp, scaleFac = 0.0;

    for (int d=0; d < nDim_; d++) {
      linres = finalNorm[d] * rhsNorm[d];
      nonlinres = realm_.l2Scaling_ * rhsNorm[d];

      if (eqSys_->firstTimeStepSolve_)
        firstNLR_[d] = nonlinres;

      tmp = std::max(std::numeric_limits<double>::epsilon(), firstNLR_[d]);
      scaledres = nonlinres / tmp;
      scaleFac += tmp * tmp;

      linearResidual_ += linres * linres;
      nonLinearResidual_ += nonlinres * nonlinres;
      scaledNonLinearResidual_ += scaledres * scaledres;
      linearSolveIterations_ += iters[d];

      if (provideOutput_) {
        const int nameOffset = eqSysName_.length() + 10;

        NaluEnv::self().naluOutputP0()
          << std::setw(nameOffset) << std::right << eqSysName_+"_"+vecNames_[d]
          << std::setw(32 - nameOffset) << std::right << iters[d] << std::setw(18)
          << std::right << linres << std::setw(15) << std::right
          << nonlinres << std::setw(14) << std::right
          << scaledres << std::endl;
      }
    }
    linearResidual_ = std::sqrt(linearResidual_);
    nonLinearResidual_ = std::sqrt(nonLinearResidual_);
    scaledNonLinearResidual_ = nonLinearResidual_ / std::sqrt(scaleFac);

    if (provideOutput_) {
      const int nameOffset = eqSysName_.length() + 8;
      NaluEnv::self().naluOutputP0()
        << std::setw(nameOffset) << std::right << eqSysName_
        << std::setw(32 - nameOffset) << std::right << linearSolveIterations_ << std::setw(18)
        << std::right << linearResidual_ << std::setw(15) << std::right
        << nonLinearResidual_ << std::setw(14) << std::right
        << scaledNonLinearResidual_ << std::endl;
    }
  }

  eqSys_->firstTimeStepSolve_ = false;

  return status;
}


void
HypreUVWLinearSystem::copy_hypre_to_stk(
  stk::mesh::FieldBase* stkField, std::vector<double>& rhsNorm)
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const auto sel = stk::mesh::selectField(*stkField)
    & meta.locally_owned_part()
    & !(stk::mesh::selectUnion(realm_.get_slave_part_vector()))
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(
    stk::topology::NODE_RANK, sel);

  std::vector<double> lclnorm(nDim_, 0.0);
  std::vector<double> gblnorm(nDim_, 0.0);
  double rhsVal = 0.0;

  for (auto b: bkts) {
    double* field = (double*) stk::mesh::field_data(*stkField, *b);
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      HypreIntType hid = get_entity_hypre_id(node);

      for (int d=0; d < nDim_; d++) {
        int sid = in * nDim_ + d;
        HYPRE_IJVectorGetValues(sln_[d], 1, &hid, &field[sid]);
        HYPRE_IJVectorGetValues(rhs_[d], 1, &hid, &rhsVal);

        lclnorm[d] += rhsVal * rhsVal;
      }
    }
  }

  stk::all_reduce_sum(bulk.parallel(), lclnorm.data(), gblnorm.data(), nDim_);

  for (int d=0; d < nDim_; d++)
    rhsNorm[d] = std::sqrt(gblnorm[d]);
}



sierra::nalu::CoeffApplier* HypreUVWLinearSystem::get_new_coeff_applier()
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
    HypreIntType maxRowID = realm_.hypreNumNodes_ * nDim_ - 1;
    
    newHostCoeffApplier.reset(new HypreUVWLinSysCoeffApplier(name_, nDim_, numPartitions, maxRowID,
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
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::HypreUVWLinSysCoeffApplier(std::string name, unsigned numDof, 
									     unsigned numPartitions, HypreIntType maxRowID,
									     HypreIntType iLower, HypreIntType iUpper,
									     HypreIntType jLower, HypreIntType jUpper,
									     Kokkos::View<HypreIntType *> partitionCount,
									     Kokkos::View<HypreIntType *> mat_count,
									     EntityToHypreIntTypeView entityToLID,
									     Kokkos::UnorderedMap<HypreIntType,HypreIntType> skippedRowsMap)
  : HypreLinSysCoeffApplier(name, numDof, numPartitions, maxRowID,
			    iLower, iUpper, jLower, jUpper,
			    partitionCount, mat_count, entityToLID, skippedRowsMap) {
  
  nDim_ = numDof;
  printf("%s %s %d : %s\n",__FILE__,__FUNCTION__,__LINE__,this->name_.c_str());
}

void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::operator()(
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

  for(unsigned i = 0; i < numEntities; ++i) {
    localIds[i] = entityToLID_[entities[i].local_offset()];
  }

  for (unsigned i=0; i < numEntities; ++i) {
    int ix = i * nDim_;
    HypreIntType hid = localIds[i];
    if (checkSkippedRows_) {
      if (skippedRowsMap_.exists(hid)) continue;
    }

    int offset = 0;
    for (unsigned k=0; k<numEntities; ++k) {
      rows_(matIndex+i*numEntities+k) = hid;
      cols_(matIndex+i*numEntities+k) = localIds[k];
      vals_(matIndex+i*numEntities+k) = lhs(ix, offset);
      offset += nDim_;
    }

    for (unsigned d=0; d < nDim_; ++d) {
      int ir = ix + d;
      rhsRows_(d,rhsIndex+i) = hid;
      rhsVals_(d,rhsIndex+i) = rhs[ir];
    }

    if ((hid >= iLower_) && (hid <= iUpper_))
      rowFilled_[hid - iLower_] = RS_FILLED;
  }
}

void
HypreUVWLinearSystem::HypreUVWLinSysCoeffApplier::applyDirichletBCs(Realm & realm, 
								    stk::mesh::FieldBase * solutionField,
								    stk::mesh::FieldBase * bcValuesField,
								    const stk::mesh::PartVector& parts) {
}

void
HypreUVWLinearSystem::buildNodeGraph(const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      count = count<0 ? 1 : count;
    }
  }

  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}


void
HypreUVWLinearSystem::buildFaceToNodeGraph(const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      count = count<0 ? (HypreIntType)(numNodes*numNodes) : count;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreUVWLinearSystem::buildEdgeToNodeGraph(const stk::mesh::PartVector& parts)
{
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      count = count<0 ? (HypreIntType)(numNodes*numNodes) : count;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreUVWLinearSystem::buildElemToNodeGraph(const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      count = count<0 ? (HypreIntType)(numNodes*numNodes) : count;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreUVWLinearSystem::buildFaceElemToNodeGraph(
  const stk::mesh::PartVector & parts)
{
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
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
      const unsigned nn = numNodes*numNodes;
      count = count<nn ? (HypreIntType)(nn) : count;
      index++;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble());
}

void
HypreUVWLinearSystem::buildReducedElemToNodeGraph(
  const stk::mesh::PartVector&)
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  beginLinearSystemConstruction();
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

void
HypreUVWLinearSystem::buildNonConformalNodeGraph(
  const stk::mesh::PartVector&)
{
  printf("%s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
  beginLinearSystemConstruction();
  printf("Done %s %s %d : name=%s\n",__FILE__,__FUNCTION__,__LINE__,name_.c_str());
}

// void
// HypreUVWLinearSystem::buildOversetNodeGraph(
//   const stk::mesh::PartVector&)
// {
//   printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
// 	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
//   beginLinearSystemConstruction();

//   // Turn on the flag that indicates this linear system has rows that must be
//   // skipped during normal sumInto process
//   hasSkippedRows_ = true;

//   // Mark all the fringe nodes as skipped so that sumInto doesn't add into these
//   // rows during assembly process
//   for(auto* oinfo: realm_.oversetManager_->oversetInfoVec_) {
//     auto node = oinfo->orphanNode_;
//     HypreIntType hid = *stk::mesh::field_data(*realm_.hypreGlobalId_, node);
//     skippedRows_.insert(hid);
//   } 
//   printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
// 	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
// }

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const stk::mesh::PartVector& parts)
{
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
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
      skippedRows_.insert(hid);
      index++;
    }
  }
  /* save these so they can be built into an UnorderedMap */
  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
}

void
HypreUVWLinearSystem::buildDirichletNodeGraph(
  const std::vector<stk::mesh::Entity>& nodeList)
{
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
  beginLinearSystemConstruction();

  // Turn on the flag that indicates this linear system has rows that must be
  // skipped during normal sumInto process
  hasSkippedRows_ = true;

  /* counter for the number elements */
  HypreIntType index=0;
  HypreIntType count=1;
  for (const auto& node: nodeList) {
    HypreIntType hid = get_entity_hypre_id(node);
    skippedRows_.insert(hid);
    index++;
  }

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());

  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
}

void 
HypreUVWLinearSystem::buildDirichletNodeGraph(const ngp::Mesh::ConnectedNodes nodeList) {
  printf("%s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
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
    skippedRows_.insert(hid);
    index++;
  }

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());

  partitionCount_.push_back(index);
  count_.push_back(count);

  printf("Done %s %s %d : name=%s, nDim=%d, numPartitions=%d, numDataPtsToAssemble=%d, skipped length=%d\n",
	 __FILE__,__FUNCTION__,__LINE__,name_.c_str(),nDim_,(int)partitionCount_.size(),(int)numDataPtsToAssemble(),(int)skippedRows_.size());
}



}  // nalu
}  // sierra

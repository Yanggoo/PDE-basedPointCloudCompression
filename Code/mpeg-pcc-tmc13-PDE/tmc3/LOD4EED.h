#pragma once

#  include "PCCTMC3Common.h"
#  include <unordered_map>
#include "EEDpredictor.h"


namespace pcc {
//---------------------------------------------------------------
void subsampleByDistanceForEED(
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& input,
  const int32_t shiftBits0,
  std::vector<uint32_t>& retained,
  std::vector<uint32_t>& indexes,
  MortonIndexMap3d& atlas);
//===============================================================================
void subsampleForEED(
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const PCCPointSet3& pointCloud,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& input,
  const int32_t lodIndex,
  std::vector<uint32_t>& retained,
  std::vector<uint32_t>& indexes,
  MortonIndexMap3d& atlas);
//===================================================================================
//==================================================================================
void computeNearestEEDNeighbors(
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  std::vector<uint32_t>& indexes,
  std::vector<EEDPCCPredictor>& EEDpredictors,
  std::vector<uint32_t>& pointIndexToEEDPredictorIndex,
  int32_t pointCount);
//===============================================================================
void buildPredictorsFastForEED(
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const PCCPointSet3& pointCloud,
  int32_t minGeomNodeSizeLog2,  //0
  int geom_num_points_minus1,
  std::vector<PCCPredictor>& predictors,
  std::vector<uint32_t>& numberOfPointsPerLevelOfDetail,
  std::vector<uint32_t>& indexes,
  std::vector<EEDPCCPredictor>& EEDpredictors);
//=========================================================================
void createGussianWeight(float& sigma, std::vector<float>& neighboursW);

} 

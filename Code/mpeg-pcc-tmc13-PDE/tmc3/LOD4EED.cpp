#pragma once

//this file contains functions about the special level of detail method designed for my prediction framework

#include "LOD4EED.h"
#include "PCCTMC3Common.h"
#include "PCCMath.h"
#include "PCCPointSet.h"
#include "constants.h"
#include "hls.h"
#include "nanoflann.hpp"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace pcc {
//===================================================================================
// This function use a MortonIndexMap3d structure to storage positions of points in local region and update them.
// In this way, the complexity will be reduced from n*n to n.
// This function splite the point indexes in input into "retained" and "indexes", enabling each point in retained
// has at least one neighbour in indexes.
void
subsampleForEED(
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const PCCPointSet3& pointCloud,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& input,
  const int32_t lodIndex,
  std::vector<uint32_t>& retained,
  std::vector<uint32_t>& indexes,
  MortonIndexMap3d& atlas)
{
  const auto shiftBits0 = 0;  //lodIndex=0
  assert(retained.empty());
  if (input.size() == 1) {
    indexes.push_back(input[0]);
    return;
  }

  const int64_t radius2 = 2ll << (shiftBits0 << 1);
  const int32_t shiftBits = shiftBits0 + 1;
  const int32_t shiftBits3 = 3 * shiftBits;
  const int32_t atlasBits = 3 * atlas.cubeSizeLog2();
  // NB: when the atlas boundary is greater than 2^63, all points belong
  //     to a single atlas.  The clipping is necessary to avoid undefined
  //     behaviour of shifts greater than or equal to the word size.
  const int32_t atlasBoundaryBit = std::min(63, shiftBits3 + atlasBits);

  // these neighbour offsets are relative to basePosition
  static const uint8_t kNeighOffset[20] = {
    7,   // { 0,  0,  0}
    3,   // {-1,  0,  0}
    5,   // { 0, -1,  0}
    6,   // { 0,  0, -1}
    12,  // { 0, -1,  1}
    10,  // {-1,  0,  1}
    17,  // {-1,  1,  0}
    20,  // { 0,  1, -1}
    34,  // { 1,  0, -1}
    33,  // { 1, -1,  0}
    4,   // { 0, -1, -1}
    2,   // {-1,  0, -1}
    1,   // {-1, -1,  0}
    24,  // {-1,  1,  1}
    40,  // { 1, -1,  1}
    48,  // { 1,  1, -1}
    32,  // { 1, -1, -1}
    16,  // {-1,  1, -1}
    8,   // {-1, -1,  1}
    0,   // {-1, -1, -1}
  };

  atlas.reserve(indexes.size() >> 1);
  int64_t curAtlasId = -1;
  int64_t lastRetainedMortonCode = -1;

  for (const auto index : input) {
    const auto& point = packedVoxel[index].position;
    const int64_t mortonCode = packedVoxel[index].mortonCode;
    const int64_t pointAtlasId = mortonCode >> atlasBoundaryBit;
    const int64_t mortonCodeShiftBits3 = mortonCode >> shiftBits3;

    if (curAtlasId != pointAtlasId) {
      atlas.clearUpdates();
      curAtlasId = pointAtlasId;
    }

    if (retained.empty()) {
      retained.push_back(index);
      lastRetainedMortonCode = mortonCodeShiftBits3;
      atlas.set(lastRetainedMortonCode, int32_t(retained.size()) - 1);
      continue;
    }

    // if (lastRetainedMortonCode == mortonCodeShiftBits3) {
    //   indexes.push_back(index);
    //   continue;
    // }

    // the position of the parent, offset by (-1,-1,-1)
    const auto basePosition = morton3dAdd(mortonCodeShiftBits3, -1ll);
    bool found = false;
    for (int32_t n = 0; n < 20 && !found; ++n) {
      const auto neighbMortonCode = morton3dAdd(basePosition, kNeighOffset[n]);
      if ((neighbMortonCode >> atlasBits) != curAtlasId)
        continue;

      const auto unit = atlas.get(neighbMortonCode);
      for (int32_t k = unit.start; k < unit.end; ++k) {
        const auto delta = (packedVoxel[retained[k]].position - point);
        if (delta.getNorm2<int64_t>() <= radius2) {
          found = true;
          break;
        }
      }
    }

    if (found) {
      indexes.push_back(index);
    } else {
      retained.push_back(index);
      lastRetainedMortonCode = mortonCodeShiftBits3;
      atlas.set(lastRetainedMortonCode, int32_t(retained.size()) - 1);
    }
  }
}


//===================================================================================
//This is a structure used to help predictor search neighbours
//It divides an ordered index sequence into segments and records the location of the last lookup
class AltasIndexMaps3D {
public:
  struct StartAndSearchIndex {
    int32_t startindex;
    int32_t searchindex;
    int32_t endindex;
  };
  void inseart(int64_t mortonCode, int32_t index)
  {
    StartAndSearchIndex tmp;
    tmp.startindex = index;
    tmp.searchindex = index;
    tmp.endindex = index;
    SearchMap[mortonCode] = tmp;
  }
  int32_t getStartIndex(int64_t mortonCode)
  {
    return SearchMap[mortonCode].startindex;
  }
  int32_t getSearchIndex(int64_t mortonCode)
  {
    return SearchMap[mortonCode].searchindex;
  }
  int32_t getEndIndex(int64_t mortonCode)
  {
    return SearchMap[mortonCode].endindex;
  }
  void setSearchIndex(int64_t mortonCode, int32_t index)
  {
    SearchMap[mortonCode].searchindex = index;
  }
  void setEndIndex(int64_t mortonCode, int32_t index)
  {
    SearchMap[mortonCode].endindex = index;
  }
  void reserve(const uint32_t sz) { SearchMap.reserve(sz); }

private:
  std::unordered_map<int64_t, StartAndSearchIndex> SearchMap;
};
//==================================================================================
//This function is responsible for searching the potential 18 neighbours of each point in the pointcloud
void
computeNearestEEDNeighbors(
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  std::vector<uint32_t>& indexes,
  std::vector<EEDPCCPredictor>& EEDpredictors,
  std::vector<uint32_t>& pointIndexToEEDPredictorIndex,
  int32_t pointCount)
{
  const int32_t log2CubeSize = 7;
  MortonIndexMap3d atlas;
  atlas.resize(log2CubeSize);
  atlas.init();
  atlas.reserve(2 << (2 * log2CubeSize) * 6);
  const int32_t atlasBits = 3 * log2CubeSize;
  const int32_t atlasBoundaryBit = std::min(63, atlasBits);
  //if point A is B's neighbour, so point B must be A's neighbour
  //Therefore, for each point, we don't need search all its neighbour
  static const uint8_t EEDkNeighOffset[13] = {
    // 0,   // {-1, -1, -1}X
    1,   // {-1, -1,  0}
    2,   // {-1,  0, -1}
    3,   // {-1,  0,  0}
    4,   // { 0, -1, -1}
    5,   // { 0, -1,  0}
    6,   // { 0,  0, -1}
    7,   // { 0,  0,  0}
         // 8,   // {-1, -1,  1}X
    10,  // {-1,  0,  1}
    12,  // { 0, -1,  1}
         //  16,  // {-1,  1, -1}X
    17,  // {-1,  1,  0}
         // 24,  // {-1,  1,  1}X
    20,  // { 0,  1, -1}
         //  32,  // { 1, -1, -1}X
    33,  // { 1, -1,  0}
    34,  // { 1,  0, -1}
         // 40,  // { 1, -1,  1}X
         // 48,  // { 1,  1, -1}X
  };

  static const uint8_t EEDNeighbourIndex[13] = {
    //0,   // {-1, -1, -1}X
    9,   // {-1, -1,  0}
    3,   // {-1,  0, -1}
    12,  // {-1,  0,  0}
    1,   // { 0, -1, -1}
    10,  // { 0, -1,  0}
    4,   // { 0,  0, -1}
    13,  // { 0,  0,  0}
    //18,  // {-1, -1,  1}X
    21,  // {-1,  0,  1}
    19,  // { 0, -1,  1}
    //6,   // {-1,  1, -1}X
    15,  // {-1,  1,  0}
    //24,  // {-1,  1,  1}X
    7,  // { 0,  1, -1}
    //2,   // { 1, -1, -1}X
    11,  // { 1, -1,  0}
    5,   // { 1,  0, -1}
         // 20,  // { 1, -1,  1}X
         // 8,   // { 1,  1, -1}X
  };

  int64_t curAtlasId = -1;
  int64_t lastMortonCodeShift3 = -1;
  int64_t cubeIndex = 0;

  AltasIndexMaps3D AtlasIdmap;
  AtlasIdmap.reserve(pointCount / (2 << (2 * log2CubeSize)));

  for (uint32_t index = 0; index < pointCount; index++) {
    const auto& pv = packedVoxel[index];
    const int64_t mortonCode = pv.mortonCode;
    const int64_t pointAtlasId = mortonCode >> atlasBoundaryBit;
    const int32_t pointIndex = pv.index;
    auto& predictor = EEDpredictors[pointIndex];
    pointIndexToEEDPredictorIndex[pointIndex] = index;

    //update MortonIndexMap3d structure when we are tranversing points
    //make sure the information in MortonIndexMap3d is what we need
    if (curAtlasId != pointAtlasId) {
      atlas.clearUpdates();
      curAtlasId = pointAtlasId;
      AtlasIdmap.inseart(curAtlasId, index);
      while (cubeIndex < pointCount
             && (packedVoxel[cubeIndex].mortonCode >> atlasBoundaryBit)
               == curAtlasId) {
        atlas.set(packedVoxel[cubeIndex].mortonCode, cubeIndex);
        ++cubeIndex;
      }
      AtlasIdmap.setEndIndex(curAtlasId, cubeIndex - 1);
    }

    const auto basePosition = morton3dAdd(mortonCode, -1ll);
    for (int32_t n = 0; n < 13; ++n) {
      const auto neighbMortonCode =
        morton3dAdd(basePosition, EEDkNeighOffset[n]);
      if ((neighbMortonCode >> atlasBits) != curAtlasId) {
        auto searchindex =
          AtlasIdmap.getSearchIndex(neighbMortonCode >> atlasBits);
        if (searchindex < 0)
          continue;
        if (packedVoxel[searchindex].mortonCode > neighbMortonCode)
          searchindex =
            AtlasIdmap.getStartIndex(neighbMortonCode >> atlasBits);
        int32_t left = searchindex;
        int32_t right = AtlasIdmap.getEndIndex(neighbMortonCode >> atlasBits);
        // use binary search
        while (left <= right) {
          searchindex = (left + right) / 2;
          if (packedVoxel[searchindex].mortonCode >= neighbMortonCode)
            right = searchindex - 1;
          else
            left = searchindex + 1;
        }
        searchindex = left;
        if (packedVoxel[searchindex].mortonCode == neighbMortonCode) {
            //record neighbours in the stucture EEDpredictors
          predictor.EEDneighbours[EEDNeighbourIndex[n]] =
            packedVoxel[searchindex].index;
          EEDpredictors[packedVoxel[searchindex].index]
            .EEDneighbours[26 - EEDNeighbourIndex[n]] = pointIndex;
          AtlasIdmap.setSearchIndex(
            neighbMortonCode >> atlasBits, searchindex);
          continue;
        }
        AtlasIdmap.setSearchIndex(
          neighbMortonCode >> atlasBits,
          std::max(
            searchindex - 1,
            AtlasIdmap.getStartIndex(neighbMortonCode >> atlasBits)));
        continue;
      }
      //record itself
      const auto range = atlas.get(neighbMortonCode);
      for (int32_t k = range.start; k < range.end; ++k) {
        predictor.EEDneighbours[EEDNeighbourIndex[n]] = packedVoxel[k].index;
        EEDpredictors[packedVoxel[k].index]
          .EEDneighbours[26 - EEDNeighbourIndex[n]] = pointIndex;
      }
    }
  }
}
//===============================================================================
//This funtion use subsampleForEED to devide layers and prepare PCCPredictor for the first layer
void
buildPredictorsFastForEED(
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const PCCPointSet3& pointCloud,
  int32_t minGeomNodeSizeLog2,  //0
  int geom_num_points_minus1,
  std::vector<PCCPredictor>& predictors,
  std::vector<uint32_t>& numberOfPointsPerLevelOfDetail,
  std::vector<uint32_t>& indexes,
  std::vector<EEDPCCPredictor>& EEDpredictors)
{
  const int32_t pointCount = int32_t(pointCloud.getPointCount());
  assert(pointCount);

  std::vector<MortonCodeWithIndex> packedVoxel;
  computeMortonCodesUnsorted(pointCloud, aps.lodNeighBias, packedVoxel);

  if (!aps.canonical_point_order_flag)
    std::sort(packedVoxel.begin(), packedVoxel.end());

  std::vector<uint32_t> retained, input, pointIndexToPredictorIndex;
  pointIndexToPredictorIndex.resize(pointCount);
  retained.reserve(pointCount);
  input.resize(pointCount);
  for (uint32_t i = 0; i < pointCount; ++i) {
    input[i] = i;
  }

  // prepare output buffers
  predictors.resize(pointCount);
  EEDpredictors.resize(pointCount);
  numberOfPointsPerLevelOfDetail.resize(0);
  indexes.resize(0);
  indexes.reserve(pointCount);
  numberOfPointsPerLevelOfDetail.reserve(2);
  numberOfPointsPerLevelOfDetail.push_back(pointCount);

  std::vector<Box3<int32_t>> bBoxes;

  const int32_t log2CubeSize = 7;
  MortonIndexMap3d atlas;
  atlas.resize(log2CubeSize);
  atlas.init();

  auto maxNumDetailLevels = aps.maxNumDetailLevels();
  //assert(maxNumDetailLevels == 2);
  if (aps.attr_encoding == AttributeEncoding::kPredictingTransform)
    maxNumDetailLevels = 2;
  int32_t predIndex = int32_t(pointCount);
  for (auto lodIndex = minGeomNodeSizeLog2;
       !input.empty() && lodIndex < maxNumDetailLevels; ++lodIndex) {
    const int32_t startIndex = indexes.size();
    if (lodIndex == maxNumDetailLevels - 1) {
      for (const auto index : input) {
        indexes.push_back(index);
      }
    } else {
      if (lodIndex == minGeomNodeSizeLog2)
        subsampleForEED(
          aps, abh, pointCloud, packedVoxel, input, lodIndex, retained,
          indexes, atlas);
      else
        subsample(
          aps, abh, pointCloud, packedVoxel, input, lodIndex, retained,
          indexes, atlas);
    }
    const int32_t endIndex = indexes.size();

    computeNearestNeighbors(
      aps, abh, packedVoxel, retained, startIndex, endIndex, lodIndex, indexes,
      predictors, pointIndexToPredictorIndex, predIndex, atlas);

    if (!retained.empty()) {
      numberOfPointsPerLevelOfDetail.push_back(retained.size());
    }
    input.resize(0);
    std::swap(retained, input);
  }
  std::reverse(indexes.begin(), indexes.end());
  updatePredictors(pointIndexToPredictorIndex, predictors);
  std::reverse(
    numberOfPointsPerLevelOfDetail.begin(),
    numberOfPointsPerLevelOfDetail.end());

  std::vector<uint32_t> pointIndexToEEDPredictorIndex;
  pointIndexToEEDPredictorIndex.resize(pointCount);
  computeNearestEEDNeighbors(
    aps, abh, packedVoxel, indexes, EEDpredictors,
    pointIndexToEEDPredictorIndex, pointCount);
}
//=========================================================================
//Create Gussian Weight, which will be used in the Gussian filting process for point cloud
void
createGussianWeight(float& sigma, std::vector<float>& neighboursW)
{
  const float neighbordis[kFixedEEDneighbourCount] = {
    3, 2, 3, 2, 1, 2, 3, 2, 3, 2, 1, 2, 1, 0,
    1, 2, 1, 2, 3, 2, 3, 2, 1, 2, 3, 2, 3};
  for (int8_t i = 0; i < kFixedEEDneighbourCount; i++)
    neighboursW[i] = exp(-neighbordis[i] / 2 / sigma / sigma);
}

}  // namespace pcc
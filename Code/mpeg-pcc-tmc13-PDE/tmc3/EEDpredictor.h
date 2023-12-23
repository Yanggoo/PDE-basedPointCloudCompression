#pragma once
#include "PCCTMC3Common.h"
using namespace pcc;

//This structure contains neighbours and local texture features for a point
//Besides that, my method needs several ierations to minimize the energy of the whole point cloud,
//which is a complex computation containing millions of points' attributes, and  those attributes are variable.
//Considering complexity, instead of using gradient descent techniques in deep learning, I choose to directly evaluate the expression for points.
//I separate constants from variables so that they can be used multiple times in each iteration.
struct EEDPCCPredictor {
  uint32_t EEDneighbours[kFixedEEDneighbourCount];
  float diffusiontensor[3][6];
  float* tensorcoffesforneighboursweight;
  uint32_t* neighbourindex;
  int32_t neighboursCount;
  EEDPCCPredictor()
  {
    memset(
      EEDneighbours, UINT32_MAX, sizeof(uint32_t) * kFixedEEDneighbourCount);
    memset(diffusiontensor, 0, sizeof(diffusiontensor));
    diffusiontensor[0][0] = diffusiontensor[0][1] = diffusiontensor[0][2] =
      diffusiontensor[1][0] = diffusiontensor[1][1] = diffusiontensor[1][2] =
        diffusiontensor[2][0] = diffusiontensor[2][1] = diffusiontensor[2][2] =
          1;
    neighboursCount = 0;
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  void freetensorcoffesforneighboursweight()
  {
    free(tensorcoffesforneighboursweight);
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  //These functions analyze the neighour occupancy for each point, which will be later used to evaluate the expression for points.
  //Due to point clouds' occupancy uncertainty, there are 2**18 occupancy situations.
  //So, I devide the complex situations into several small situations.
  //I anaylyze only 4 points in a local region at a time.
  void CalUXYZ0(
    const int32_t Utemplate[4][4],
    float distance,
    int32_t occupycode,
    size_t number,
    size_t dim1,
    size_t dim2,
    uint32_t index2pos[27])
  {
    size_t i = 0;
    float disaddone2 = pow(distance + 1, 2) / 32;
    float dis2 = pow(distance, 2) / 32;
    float nodis = 1 / 64;
    float diss = pow(distance + 1, 2) / 64 + pow(distance, 2) / 64;
    switch (occupycode) {
    case 15: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[4] = {m0pos, m1pos, m2pos, m3pos};
      for (size_t j = 0; j < 4; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += -disaddone2;
      }
      break;
    }
    case 7: {
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m1pos, m2pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
      }
      break;
    }

    case 11: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m0pos, m2pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += -diss;
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += nodis;
      }
      break;
    }
    case 13: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m0pos, m1pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += -diss;
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += nodis;
      }
      break;
    }
    case 9: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[2] = {m0pos, m3pos};
      for (size_t j = 0; j < 2; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += -dis2;
      }
      break;
    }
    }
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  void CalUXYZ1(
    const int32_t Utemplate[4][4],
    float distance,
    int32_t occupycode,
    size_t number,
    size_t dim1,
    size_t dim2,
    uint32_t index2pos[27])
  {
    size_t i = 1;
    float disaddone2 = pow(distance + 1, 2) / 32;
    float dis2 = pow(distance, 2) / 32;
    float nodis = 1 / 64;
    float diss = pow(distance + 1, 2) / 64 + pow(distance, 2) / 64;
    switch (occupycode) {
    case 15: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[4] = {m0pos, m1pos, m2pos, m3pos};
      for (size_t j = 0; j < 4; j++) {
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += disaddone2;
      }
      break;
    }
    case 7: {
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m1pos, m2pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += diss;
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
      }
      break;
    }

    case 11: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m0pos, m2pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += nodis;
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += nodis;
      }
      break;
    }
    case 14: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      uint32_t coffindex[3] = {m0pos, m1pos, m2pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += diss;
      }
      break;
    }
    case 6: {
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      uint32_t coffindex[2] = {m1pos, m2pos};
      for (size_t j = 0; j < 2; j++) {
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += dis2;
      }
      break;
    }
    }
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  void CalUXYZ2(
    const int32_t Utemplate[4][4],
    float distance,
    int32_t occupycode,
    size_t number,
    size_t dim1,
    size_t dim2,
    uint32_t index2pos[27])
  {
    size_t i = 2;
    float disaddone2 = pow(distance + 1, 2) / 32;
    float dis2 = pow(distance, 2) / 32;
    float nodis = 1 / 64;
    float diss = pow(distance + 1, 2) / 64 + pow(distance, 2) / 64;
    switch (occupycode) {
    case 15: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[4] = {m0pos, m1pos, m2pos, m3pos};
      for (size_t j = 0; j < 4; j++) {
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += disaddone2;
      }
      break;
    }
    case 7: {
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m1pos, m2pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += diss;
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
      }
      break;
    }

    case 13: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m0pos, m1pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += nodis;
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += nodis;
      }
      break;
    }
    case 14: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      uint32_t coffindex[3] = {m0pos, m1pos, m2pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m0pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += diss;
      }
      break;
    }
    case 6: {
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      uint32_t coffindex[2] = {m1pos, m2pos};
      for (size_t j = 0; j < 2; j++) {
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += dis2;
      }
      break;
    }
    }
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  void CalUXYZ3(
    const int32_t Utemplate[4][4],
    float distance,
    int32_t occupycode,
    size_t number,
    size_t dim1,
    size_t dim2,
    uint32_t index2pos[27])
  {
    size_t i = 3;
    float disaddone2 = pow(distance + 1, 2) / 32;
    float dis2 = pow(distance, 2) / 32;
    float nodis = 1 / 64;
    float diss = pow(distance + 1, 2) / 64 + pow(distance, 2) / 64;
    switch (occupycode) {
    case 15: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[4] = {m0pos, m1pos, m2pos, m3pos};
      for (size_t j = 0; j < 4; j++) {
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += -disaddone2;
      }
      break;
    }
    case 11: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m2pos = index2pos[Utemplate[i][2]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m0pos, m2pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += nodis;
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += -diss;
      }
      break;
    }

    case 13: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[3] = {m0pos, m1pos, m3pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += nodis;
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += -diss;
      }
      break;
    }
    case 14: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m1pos = index2pos[Utemplate[i][1]];
      auto m2pos = index2pos[Utemplate[i][2]];
      uint32_t coffindex[3] = {m0pos, m1pos, m2pos};
      for (size_t j = 0; j < 3; j++) {
        tensorcoffesforneighboursweight
          [m1pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
        tensorcoffesforneighboursweight
          [m2pos * dim1 + coffindex[j] * dim2 + number] += -nodis;
      }
      break;
    }
    case 9: {
      auto m0pos = index2pos[Utemplate[i][0]];
      auto m3pos = index2pos[Utemplate[i][3]];
      uint32_t coffindex[2] = {m0pos, m3pos};
      for (size_t j = 0; j < 2; j++) {
        tensorcoffesforneighboursweight
          [m3pos * dim1 + coffindex[j] * dim2 + number] += -dis2;
      }
      break;
    }
    }
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  //This function use functions above to evaluate the expression for points.
  void Caltensorcoffesforneighboursweight(
    float diffX[27],
    float diffY[27],
    float diffZ[27],
    const int32_t UXUYtemplate[4][4],
    const int32_t UXUZtemplate[4][4],
    const int32_t UYUZtemplate[4][4],
    const float distance)
  {
    uint32_t index2pos[27];
    memset(index2pos, -1, sizeof(uint32_t) * 27);
    for (size_t i = 0; i < 27; i++) {
      if (EEDneighbours[i] != UINT32_MAX) {
        index2pos[i] = neighboursCount;
        neighboursCount++;
      }
    }
    neighbourindex = (uint32_t*)malloc(sizeof(uint32_t) * neighboursCount);
    for (size_t i = 0; i < 27; i++) {
      if (index2pos[i] != -1) {
        neighbourindex[index2pos[i]] = i;
      }
    }
    tensorcoffesforneighboursweight =
      (float*)malloc(sizeof(float) * neighboursCount * neighboursCount * 6);
    memset(
      tensorcoffesforneighboursweight, 0,
      sizeof(float) * neighboursCount * neighboursCount * 6);
    size_t dim1 = neighboursCount * 6;
    size_t dim2 = 6;
    auto thispos = index2pos[13];
    for (size_t i = 0; i < neighboursCount; i++) {
      auto index = neighbourindex[i];
      if (index == 13)
        continue;
      //UX2
      tensorcoffesforneighboursweight[i * dim1 + i * dim2 + 0] +=
        pow(diffX[index], 2) / 10;
      tensorcoffesforneighboursweight[i * dim1 + thispos * dim2 + 0] +=
        pow(diffX[index], 2) / 10;
      //UY2
      tensorcoffesforneighboursweight[i * dim1 + i * dim2 + 1] +=
        pow(diffY[index], 2) / 10;
      tensorcoffesforneighboursweight[i * dim1 + thispos * dim2 + 1] +=
        pow(diffY[index], 2) / 10;
      //UZ2
      tensorcoffesforneighboursweight[i * dim1 + i * dim2 + 2] +=
        pow(diffZ[index], 2) / 10;
      tensorcoffesforneighboursweight[i * dim1 + thispos * dim2 + 2] +=
        pow(diffZ[index], 2) / 10;
    }
    for (size_t i = 0; i < 4; i++) {
      int32_t occupycodeXY, occupycodeXZ, occupycodeYZ;
      occupycodeXY = occupycodeXZ = occupycodeYZ = 0;
      for (size_t j = 0; j < 4; j++) {
        if (EEDneighbours[UXUYtemplate[i][j]] != UINT32_MAX) {
          occupycodeXY = (occupycodeXY << 1) + 1;
        } else {
          occupycodeXY = occupycodeXY << 1;
        }
        if (EEDneighbours[UXUZtemplate[i][j]] != UINT32_MAX) {
          occupycodeXZ = (occupycodeXZ << 1) + 1;
        } else {
          occupycodeXZ = occupycodeXZ << 1;
        }
        if (EEDneighbours[UYUZtemplate[i][j]] != UINT32_MAX) {
          occupycodeYZ = (occupycodeYZ << 1) + 1;
        } else {
          occupycodeYZ = occupycodeYZ << 1;
        }
      }
      switch (i) {
      case 0: {
        CalUXYZ0(
          UXUYtemplate, distance, occupycodeXY, 3, dim1, dim2, index2pos);
        CalUXYZ0(
          UXUZtemplate, distance, occupycodeXZ, 4, dim1, dim2, index2pos);
        CalUXYZ0(
          UYUZtemplate, distance, occupycodeYZ, 5, dim1, dim2, index2pos);
        break;
      }
      case 1: {
        CalUXYZ1(
          UXUYtemplate, distance, occupycodeXY, 3, dim1, dim2, index2pos);
        CalUXYZ1(
          UXUZtemplate, distance, occupycodeXZ, 4, dim1, dim2, index2pos);
        CalUXYZ1(
          UYUZtemplate, distance, occupycodeYZ, 5, dim1, dim2, index2pos);
        break;
      }
      case 2: {
        CalUXYZ2(
          UXUYtemplate, distance, occupycodeXY, 3, dim1, dim2, index2pos);
        CalUXYZ2(
          UXUZtemplate, distance, occupycodeXZ, 4, dim1, dim2, index2pos);
        CalUXYZ2(
          UYUZtemplate, distance, occupycodeYZ, 5, dim1, dim2, index2pos);
        break;
      }
      case 3: {
        CalUXYZ3(
          UXUYtemplate, distance, occupycodeXY, 3, dim1, dim2, index2pos);
        CalUXYZ3(
          UXUZtemplate, distance, occupycodeXZ, 4, dim1, dim2, index2pos);
        CalUXYZ3(
          UYUZtemplate, distance, occupycodeYZ, 5, dim1, dim2, index2pos);
        break;
      }
      }
    }
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  //Get the prediction of points' attributes in each iteration to minimize the energy of point cloud step by step.
  Vec3<attr_t> predictColorFast(
    const PCCPointSet3& EEDpointCloud,
    std::vector<EEDPCCPredictor>& EEDpredictors)
  {
    Vec3<float> predicted{0};
    size_t dim1 = neighboursCount * 6;
    size_t dim2 = 6;
    float* neighboursweightY = (float*)malloc(sizeof(float) * neighboursCount);
    memset(neighboursweightY, 0, sizeof(float) * neighboursCount);
    float* neighboursweightU = (float*)malloc(sizeof(float) * neighboursCount);
    memset(neighboursweightU, 0, sizeof(float) * neighboursCount);
    float* neighboursweightV = (float*)malloc(sizeof(float) * neighboursCount);
    memset(neighboursweightV, 0, sizeof(float) * neighboursCount);
    float neighboursweightsumY, neighboursweightsumU, neighboursweightsumV;
    neighboursweightsumY = neighboursweightsumU = neighboursweightsumV = 0;
    for (size_t i = 0; i < neighboursCount; i++) {
      auto idx1 = EEDneighbours[neighbourindex[i]];
      for (size_t j = 0; j < neighboursCount; j++) {
        auto idx2 = EEDneighbours[neighbourindex[j]];
        auto neighbourdiffusiontensor = EEDpredictors[idx2].diffusiontensor;
        for (size_t k = 0; k < 6; k++) {
          if (tensorcoffesforneighboursweight[i * dim1 + j * dim2 + k] != 0) {
            neighboursweightY[i] +=
              tensorcoffesforneighboursweight[i * dim1 + j * dim2 + k]
              * neighbourdiffusiontensor[0][k];
            neighboursweightU[i] +=
              tensorcoffesforneighboursweight[i * dim1 + j * dim2 + k]
              * neighbourdiffusiontensor[1][k];
            neighboursweightV[i] +=
              tensorcoffesforneighboursweight[i * dim1 + j * dim2 + k]
              * neighbourdiffusiontensor[2][k];
          }
        }
      }
      neighboursweightsumY += neighboursweightY[i];
      neighboursweightsumU += neighboursweightU[i];
      neighboursweightsumV += neighboursweightV[i];
      auto color = EEDpointCloud.getColor(idx1);
      predicted[0] += color[0] * neighboursweightY[i];
      predicted[1] += color[1] * neighboursweightU[i];
      predicted[2] += color[2] * neighboursweightV[i];
    }
    predicted[0] = int64_t(predicted[0] / neighboursweightsumY + 0.5);
    predicted[1] = int64_t(predicted[1] / neighboursweightsumU + 0.5);
    predicted[2] = int64_t(predicted[2] / neighboursweightsumV + 0.5);
    predicted[0] = PCCClip(predicted[0], 0, 255);
    predicted[1] = PCCClip(predicted[1], 1, 511);
    predicted[2] = PCCClip(predicted[2], 1, 511);
    free(neighboursweightY);
    free(neighboursweightU);
    free(neighboursweightV);
    return Vec3<attr_t>(predicted[0], predicted[1], predicted[2]);
  }
  //---------------------------------------------------------------------------------------------------------------------------------
  //Use Gaussian filter to process point cloud
  void gussianfilter(
    const PCCPointSet3& EEDpointCloud,
    const std::vector<float> neighboursW,
    PCCPointSet3& GussianEEDpointCloud)
  {
    float weightsum = 0;
    float colorsum1, colorsum2, colorsum3;
    colorsum1 = colorsum2 = colorsum3 = 0;
    for (size_t i = 0; i < 27; i++) {
      if (EEDneighbours[i] != UINT32_MAX) {
        weightsum += neighboursW[i];
        Vec3<attr_t> neighbourColors =
          EEDpointCloud.getColor(EEDneighbours[i]);
        colorsum1 += neighboursW[i] * neighbourColors[0];
        colorsum2 += neighboursW[i] * neighbourColors[1];
        colorsum3 += neighboursW[i] * neighbourColors[2];
      }
    }
    Vec3<attr_t> gussianColor;
    gussianColor[0] = colorsum1 / weightsum;
    gussianColor[1] = colorsum2 / weightsum;
    gussianColor[2] = colorsum3 / weightsum;
    GussianEEDpointCloud.setColor(EEDneighbours[13], gussianColor);
  }
  //--------------------------------------------------------------------------------------------------------------------------------
  //Use attributes updated in iterations to recalculate local texture features.
  void updateDiffusionTensor(
    const PCCPointSet3& EEDpointCloud,
    const PCCPointSet3& GussianEEDpointCloud,
    float diffX[27],
    float diffY[27],
    float diffZ[27],
    float lambda,
    bool* isseed,
    bool* ispredicted,
    float seedweight,
    float predictedweight)
  {
    for (size_t r = 0; r < 3; r++) {
      float deltaX, deltaY, deltaZ;
      deltaX = deltaY = deltaZ = 0;
      int32_t neighborCount = -1;
      int32_t neighborCountX, neighborCountY, neighborCountZ;
      neighborCountX = neighborCountY = neighborCountZ = 0;
      Vec3<attr_t> thisColor =
        GussianEEDpointCloud.getColor(EEDneighbours[13]);
      for (size_t i = 0; i < 27; i++) {
        if (EEDneighbours[i] != UINT32_MAX) {
          neighborCount++;
          Vec3<attr_t> neighbourColor =
            GussianEEDpointCloud.getColor(EEDneighbours[i]);
          float diff = neighbourColor[r] - thisColor[r];
          deltaX += diffX[i] * diff;
          deltaY += diffY[i] * diff;
          deltaZ += diffZ[i] * diff;
          if (diffX[i] != 0)
            neighborCountX++;
          if (diffY[i] != 0)
            neighborCountY++;
          if (diffZ[i] != 0)
            neighborCountZ++;
        }
      }
      float deltasize = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
      if (neighborCount != 0 || deltasize != 0) {
        deltaX = neighborCountX == 0 ? 0 : deltaX / (neighborCountX);
        deltaY = neighborCountY == 0 ? 0 : deltaY / (neighborCountY);
        deltaZ = neighborCountZ == 0 ? 0 : deltaZ / (neighborCountZ);
        deltasize = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
        // float maindirectionscale = (deltasize == 0)
        //   ? 0
        //   : ((1 - exp(-3.31488 * pow(lambda, 4) / pow(deltasize, 4))) - 1)
        //     / deltasize;
        float maindirectionscale = (deltasize == 0)
          ? 0
          : (sqrt(1 / (1 + deltasize / lambda / lambda)) - 1) / deltasize;
        diffusiontensor[r][0] = 1 + maindirectionscale * deltaX * deltaX;
        diffusiontensor[r][1] = 1 + maindirectionscale * deltaY * deltaY;
        diffusiontensor[r][2] = 1 + maindirectionscale * deltaZ * deltaZ;
        diffusiontensor[r][3] = maindirectionscale * deltaX * deltaY;
        diffusiontensor[r][4] = maindirectionscale * deltaX * deltaZ;
        diffusiontensor[r][5] = maindirectionscale * deltaY * deltaZ;
      } else {
        diffusiontensor[r][0] = diffusiontensor[r][1] = diffusiontensor[r][2] =
          1;
        diffusiontensor[r][3] = diffusiontensor[r][4] = diffusiontensor[r][5] =
          0;
      }
      if (isseed[EEDneighbours[13]]) {
        for (size_t i = 0; i < 6; i++)
          diffusiontensor[r][i] *= seedweight;
      }
      if (ispredicted[EEDneighbours[13]]) {
        for (size_t i = 0; i < 6; i++)
          diffusiontensor[r][i] *= predictedweight;
      }
    }
  }
};
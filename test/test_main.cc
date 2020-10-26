// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// gDLS*: Generalized Pose-and-Scale Estimation Given Scale and Gravity Priors
//
// Victor Fragoso, Joseph DeGol, Gang Hua.
// Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition 2020.
//
// Please contact the author of this library if you have any questions.
// Author: Victor Fragoso (victor.fragoso@microsoft.com)

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "gtest/gtest.h"

int main(int argc, char* argv[]) {
  // Initialize glog.
  google::InitGoogleLogging(argv[0]);
  // Initialize gtest.
  ::testing::InitGoogleTest(&argc, argv);
  // Parse flags using gflags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Running tests with logs enabled.";
  return RUN_ALL_TESTS();
}

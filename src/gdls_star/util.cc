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

#include "gdls_star/util.h"

#include <vector>
#include <Eigen/Core>
#include "gdls_star/camera_feature_correspondence_2d_3d.h"
#include "gdls_star/gdls_star.h"

namespace msft {

GdlsStar::Input ComputeInputDatum(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences) {
  GdlsStar::Input input;
  input.ray_origins.resize(correspondences.size());
  input.ray_directions.resize(correspondences.size());
  input.world_points.resize(correspondences.size());
  for (int i = 0; i < correspondences.size(); ++i) {
    // Keep 3D point.
    input.world_points[i] = correspondences[i].point;
    // Compute ray direction.
    const Eigen::Vector2d& pixel = correspondences[i].observation;
    input.ray_directions[i] = correspondences[i].camera.PixelToUnitRay(pixel);
    // Compute ray origins.
    input.ray_origins[i] = correspondences[i].camera.GetPosition();
  }
  return input;
}

}  // namespace msft

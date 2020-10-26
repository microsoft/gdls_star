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

#ifndef GDLS_STAR_CAMERA_FEATURE_CORRESPONDENCE_2D_3D_H_
#define GDLS_STAR_CAMERA_FEATURE_CORRESPONDENCE_2D_3D_H_

#include <Eigen/Core>
#include "gdls_star/pinhole_camera.h"

namespace msft {

// The goal of this structure is to hold the 2D-3D correspondence as well as
// the individual camera observing the 3D point. This is necessary to build
// the input to gDLS* as it uses a generalized camera model.
struct CameraFeatureCorrespondence2D3D {
  // The pinhole camera seeing 2D feature. The camera pose must be wrt to the
  // generalized camera coordinate system (or query coordinate system).
  PinholeCamera camera;

  // Observed keypoint or 2D feature.
  Eigen::Vector2d observation;

  // Corresponding 3D point. This point is described wrt to the world coordinate
  // system.
  Eigen::Vector3d point;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

}  // namespace msft

#endif  // GDLS_STAR_CAMERA_FEATURE_CORRESPONDENCE_2D_3D_H_

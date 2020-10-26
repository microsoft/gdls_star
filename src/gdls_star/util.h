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

#ifndef GDLS_STAR_UTIL_H_
#define GDLS_STAR_UTIL_H_

#include <vector>
#include "gdls_star/camera_feature_correspondence_2d_3d.h"
#include "gdls_star/gdls_star.h"

namespace msft {

// Computes the gDLS* input from the 2D-3D correspondences and its respective
// camera observing the 3D point.
GdlsStar::Input ComputeInputDatum(
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences);

}  // namespace msft

#endif  // GDLS_STAR_UTIL_H_

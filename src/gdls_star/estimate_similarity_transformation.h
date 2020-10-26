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

#ifndef GDLS_STAR_ESTIMATE_SIMILARITY_TRANSFORMATION_H_
#define GDLS_STAR_ESTIMATE_SIMILARITY_TRANSFORMATION_H_

#include <Eigen/Core>
#include "gdls_star/camera_feature_correspondence_2d_3d.h"
#include "gdls_star/gdls_star.h"
#include "gdls_star/gdls_star_robust_estimator.h"

namespace msft {

// Computes the similarity transformation given 2D-3D correspondences.
// The 2D-3D correspondences contains also the camera that observes the
// projection of the candidate 3D point. This function calls the gDLS*
// directly. Thus, this function can use the gDLS* estimator as minimal
// or non-minimal solver.
//
// Params:
//   priors  gDLS* priors (scale and gravity).
//   correspondences  The 2D-3D correspondences.
GdlsStar::Solution EstimateSimilarityTransformation(
    const GdlsStar::Priors& priors,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences);

// Computes the similarity transformation using a RANSAC estimation framework.
// The function expects the ransac parameters, priors, 2D-3D correspondences,
// and the ransac_summary pointer to store RANSAC statistics. The function
// returns the estimated solution with the largest number of inliers.
//
// Params:
//   ransac_params  Ransac parameters.
//   priors  gDLS* priors (scale and gravity).
//   correspondences  The 2D-3D correspondences.
//   ransac_summary  Ransac summary.
GdlsStar::Solution EstimateSimilarityTransformation(
    const GdlsStarRobustEstimator::RansacParameters& ransac_params,
    const GdlsStar::Priors& priors,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
    GdlsStarRobustEstimator::RansacSummary* ransac_summary);

}  // namespace msft

#endif  // GDLS_STAR_ESTIMATE_SIMILARITY_TRANSFORMATION_H_

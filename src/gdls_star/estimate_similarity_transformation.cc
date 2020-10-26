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

#include "gdls_star/estimate_similarity_transformation.h"

#include <vector>
#include "gdls_star/camera_feature_correspondence_2d_3d.h"
#include "gdls_star/gdls_star.h"
#include "gdls_star/gdls_star_robust_estimator.h"
#include "gdls_star/util.h"

namespace msft {

// Computes the similarity transformation given 2D-3D correspondences and prios.
GdlsStar::Solution EstimateSimilarityTransformation(
    const GdlsStar::Priors& priors,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences) {
  GdlsStar::Solution solution;
  // Compute input datum.
  const GdlsStar::Input input = ComputeInputDatum(correspondences);
  GdlsStar estimator;
  estimator.EstimateSimilarityTransformation(input, &solution);  
  return solution;
}

// Computes the similarity transformation given 2D-3D correspondences and priors
// using a RANSAC estimator.
GdlsStar::Solution EstimateSimilarityTransformation(
    const GdlsStarRobustEstimator::RansacParameters& params,
    const GdlsStar::Priors& priors,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
    GdlsStarRobustEstimator::RansacSummary* ransac_summary) {
  GdlsStarRobustEstimator estimator(params);
  const GdlsStar::Solution solution =
      estimator.Estimate(priors, correspondences, ransac_summary);
  return solution;
}

}  // namespace msft

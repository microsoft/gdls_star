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

#ifndef GDLS_STAR_GDLS_STAR_ROBUST_ESTIMATOR_H_
#define GDLS_STAR_GDLS_STAR_ROBUST_ESTIMATOR_H_

#include <random>
#include <vector>
#include "gdls_star/gdls_star.h"

namespace msft {

// Forward declaration.
struct CameraFeatureCorrespondence2D3D;

// This class implements gDLS* wrapped in a RANSAC robust estimation framework.
// The class estimates a similarity transformation by calling gDLS* in a
// hypothesis-and-test loop. The estimator expects to receive the gDLS* priors,
// the 2D-3D correspondences including the cameras observing the points, and
// RANSAC parameters. To use this estimator, first create an instance of
// RansacParameters, set the desired parameters, and then construct a
// GdlsStarRobustEstimator instance. Finally, call the Estimate method to
// launch the robust estimation process.
//
// Example:
//
//  GdlsStarRobustEstimator::RansacParameters ransac_params;
//  ransac_params.max_iterations = 500;
//  GdlsStarRobustEstimator::RansacSummary ransac_summary;
//  GdlsStarRobustEstimator estimator(ransac_params);
//  const GdlsStar::Solution solution = estimator.Estimate(
//               priors,
//               correspondences,
//               &ransac_summary);
//
class GdlsStarRobustEstimator {
 public:
  // This structure contains the most common RANSAC parameters.
  struct RansacParameters {
    // The failure probability of RANSAC. This is useful for estimating the
    // number of iterations in RANSAC adaptively. Setting this probability
    // to 0.01 is equivalent to expect that there is a 1% chance of missing
    // the correct estimate given a minimal sample.
    double failure_probability = 0.01;

    // Reprojection error threshold (in pixels).
    double reprojection_error_thresh = 2.0;

    // Minimum number of iterations.
    int min_iterations = 100;

    // Maximum number of iterations.
    int max_iterations = 1000;

    // Random seed.
    size_t seed = 67;
  };

  // This structure contains statistics about the RANSAC run as well as
  // the set of found inliers.
  struct RansacSummary {
    // Inlier indices.
    std::vector<int> inliers;

    // Number of iterations performed in the RANSAC estimation process.
    int num_iterations = 0;

    // The confidence in the solution.
    double confidence = 0.0;

    // Number of evaluated hypotheses.
    int num_hypotheses = 0;
  };

  explicit GdlsStarRobustEstimator(const RansacParameters& params);
  ~GdlsStarRobustEstimator() = default;

  // Estimates the similarity transformation using gDLS* as a minimal solver.
  GdlsStar::Solution Estimate(
      const GdlsStar::Priors& priors,
      const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
      RansacSummary* ransac_summary);

 private:
  // Ransac parameters.
  const RansacParameters params_;

  // Pseudo random number generator.
  std::mt19937 prng_;

  // gDLS* estimator.
  GdlsStar estimator_;

  // Correspondence indices.
  std::vector<int> correspondence_indices_;

  // Helper functions.
  // Computes a random minimal sample. This is used to generate hypotheses.
  std::vector<CameraFeatureCorrespondence2D3D> Sample(
      const std::vector<CameraFeatureCorrespondence2D3D>& correspondences);

  // Regnerates random integer within a specific range.
  int RandInt(const int min_value, const int max_value);

  // Computes maximum number of iterations as a function of inlier ratio and
  // probability of failure. This functions operates as follows: given the
  // current inlier_ratio, and the probability of failure, the number of
  // iterations that are required to find a good hypothesis is
  //
  //  num_iterations = log_failure_prob / log(1 - inlier_ratio^min_sample),
  //
  // where min_sample is the minimum sample size to produce a hypothesis.
  // For more information, please see
  // https://en.wikipedia.org/wiki/Random_sample_consensus.
  int ComputeMaxIterations(const double inlier_ratio,
                           const double log_failure_prob);

  // Updates the best solution by identifying inliers and keeping the solution
  // that has the largest number of inliers.
  double UpdateBestSolution(
      const std::vector<CameraFeatureCorrespondence2D3D>& correspondences,
      const GdlsStar::Solution& estimated_solns,
      GdlsStar::Solution* best_solution,
      std::vector<int>* best_inlier_idxs);
};

}  // namespace msft

#endif  // GDLS_STAR_GDLS_STAR_ROBUST_ESTIMATOR_H_

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

#ifndef GDLS_STAR_GDLS_STAR_H_
#define GDLS_STAR_GDLS_STAR_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "math/alignment.h"

namespace msft {

// This class estimates similarity transformations from 2D-3D correspondences
// and gravity and scale priors. The implemented solver is based on
//
// V. Fragoso, J. DeGol, and G. Hua. gDLS*: Generalized Pose-and-Scale
// Estimation Given Scale and Gravity Priors. CVPR 2020.
//
// gDLS* enhances gDLS+++ by adding gravity and scale prior constraints. The
// priors are added to the gDLS+++ cost function as follows:
//
// Gravity prior = || gravity_dir_gcam x R_world_to_cam * gravity_world||^2
//
// and
//
// Scale prior = (scale_prior - estimated scale)^2.
//
// The paper shows that these additional terms still comply with the quadratic
// form of the gDLS+++ cost function (see CostParameters structure).
// Because of this, gDLS* can re-use the polynomial solver used by gDLS+++.
class GdlsStar {
  // Useful aliases for data types.
  using Matrix10d = Eigen::Matrix<double, 10, 10>;
  using Matrix3x10d = Eigen::Matrix<double, 3, 10>;
  using RowVector10d = Eigen::Matrix<double, 1, 10>;
  using RowMajorMatrixXd =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector10d = Eigen::Matrix<double, 10, 1>;

 public:
  // The cost function of gDLS* can be rewritten as follows:
  //
  // J(R) = vec(R)' * quadratic_penatly_mat * vec(R) +
  //        2 * linear_penalty_vector' * vec(R) + gamma,
  //
  // where vec(R) is the vector shown in Eq. XX of the Upnp paper.
  // The parameters of the quadratic function above are the
  //   1. The quadratic penalty matrix.
  //   2. The linear penalty vector; and
  //   3. A scalar penalty term (i.e., gamma).
  struct CostParameters {
    CostParameters() {
      // The quadratic penalty matrix.
      quadratic_penalty_mat.setZero();
      // The linear penalty matrix.
      linear_penalty_vector.setZero();
      // The constant term.
      gamma = 0.0;
    }
    ~CostParameters() = default;

    Matrix10d quadratic_penalty_mat;
    Vector10d linear_penalty_vector;
    double gamma;
  };

  // This structure holds the scale and gravity priors. The structure holds the
  // scale and gravity priors, as well as the scale and gravity penalties.
  // The gravity priors involve three params:
  //
  //  1. World down direction vector (i.e., gravity direction in the world).
  //  2. Query down direction vector (i.e., gravity direction in the general
  //     camera coordinate system).
  //  3. Gravity penalty factor for the cost function.
  //
  // The scale priors involve two parameters:
  //  1. Scale prior.
  //  2. Scale penalty.
  //
  // Note that when both penalties are zero, gDLS* defaults to gDLS+++.
  struct Priors {
    Priors() : scale_penalty(0.0),
               scale_prior(1.0),
               gravity_penalty(0.0) {
      world_down_direction.setZero();
      query_down_direction.setZero();
    }
    ~Priors() = default;

    // The down-unit direction or gravity vector for the world or reference
    // point cloud.
    Eigen::Vector3d world_down_direction;
    // The down-unit direction or gravity vector for the query non-central
    // camera.
    Eigen::Vector3d query_down_direction;
    // Scale penalty.
    double scale_penalty;
    // Scale prior.
    double scale_prior;
    // Gravity penalty.
    double gravity_penalty;
  };

  // This structure aims to collect the input for gDLS*. The input mainly aims
  // to encode the 2D-3D correspondences and the input includes:
  //   1. Ray origins (camera positions).
  //   2. Ray directions (direction vector from ray origin to a 3D point).
  //   3. World point (3D point in the world).
  //   4. Priors (see above).
  // Note that ray origin and direction are referenced wrt the generalized
  // camera coordinate system.
  struct Input {
    std::vector<Eigen::Vector3d> ray_origins;
    std::vector<Eigen::Vector3d> ray_directions;
    std::vector<Eigen::Vector3d> world_points;
    Priors priors;
  };

  // This structure encodes all the similarity transformations that gDLS*
  // computes. The structure include the rotations, translations, and scales.
  struct Solution {
    std::vector<Eigen::Quaterniond> rotations;
    std::vector<Eigen::Vector3d> translations;
    std::vector<double> scales;
  };

  // Default constructor and destructor.
  GdlsStar() = default;
  ~GdlsStar() = default;

  // Estimates the similarity transformations from the given 2D-3D
  // correspondences and priors.
  //
  // Params:
  //   input  The 2D-3D correspondences and priors.
  //   solution  The structure holding all the solutions found.
  bool EstimateSimilarityTransformation(const Input& input, Solution* solution);

 private:
  // Template matrix for the polynomial solver.
  RowMajorMatrixXd template_matrix_;

  // Helper metrices.
  struct HelperMatrices {
    Eigen::Matrix4d h_matrix;
    RowVector10d scale_factor;
    Matrix3x10d translation_factor;
    Eigen::MatrixXd f_matrix;
  };

  // Helper matrices that allows us to compute the cost parameters.
  HelperMatrices helper_matrices_;

  // Cost function parameters.
  CostParameters cost_params_;

  // Helper functions:
  // This function computes the cost parameters of the least squares function.
  // This function is exposed for testing purposes. Returns the computed cost
  // parameters, but this function also sets the cost_params_ class member.
  CostParameters ComputeCostParameters(const Input& input);

  // This function computes the rotations from the cost parameters.
  std::vector<Eigen::Quaterniond> EstimateRotations();

  // This function computes several helper matrices (e.g., h_matrix,
  // f_matrix, etc.).
  void ComputeHelperMatrices(const Input& input);

  // Computes the quadratic cost parameters.
  void ComputeLeastSquaresCostParameters(const Input& input);
};

}  // namespace msft

#endif  // GDLS_STAR_GDLS_STAR_H_

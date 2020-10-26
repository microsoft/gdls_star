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

#include "gdls_star/gdls_star.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include "math/alignment.h"
#include "math/utils.h"
#include "upnp/build_upnp_action_matrix_using_symmetry.h"

namespace msft {
namespace {
// Useful aliases.
using Matrix3x10d = Eigen::Matrix<double, 3, 10>;
using Matrix10d = Eigen::Matrix<double, 10, 10>;
using Matrix8cd = Eigen::Matrix<std::complex<double>, 8, 8>;
using Matrix8d = Eigen::Matrix<double, 8, 8>;
using RowVector10d = Eigen::Matrix<double, 1, 10>;
using Vector10d = Eigen::Matrix<double, 10, 1>;

using CostParameters = GdlsStar::CostParameters;
using Input = GdlsStar::Input;
using Priors = GdlsStar::Priors;
using Solution = GdlsStar::Solution;

constexpr int kNumMaxRotationsExploitingSymmetry = 8;

// Computes the skew-symmetric matrix to compute the cross product.
inline Eigen::Matrix3d
ComputeSkewSymmetricMatrix(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d skew_symmetric_matrix;
  skew_symmetric_matrix.setZero();
  skew_symmetric_matrix(0, 1) = -vector.z();
  skew_symmetric_matrix(1, 0) = vector.z();
  skew_symmetric_matrix(0, 2) = vector.y();
  skew_symmetric_matrix(2, 0) = -vector.y();
  skew_symmetric_matrix(1, 2) = -vector.x();
  skew_symmetric_matrix(2, 1) = vector.x();
  return skew_symmetric_matrix;
}

// This function arranges a 3D point into a 3x10 matrix so that we can
// rotate a point using the rotation expressed as a function of the monomials.
// See Eq. xxiv of the gDLS* supplemental material.
Matrix3x10d LeftMultiply(const Eigen::Vector3d& point) {
  Matrix3x10d phi_mat;
  // Row 0.
  phi_mat(0, 0) = point.x();
  phi_mat(0, 1) = point.x();
  phi_mat(0, 2) = -point.x();
  phi_mat(0, 3) = -point.x();
  phi_mat(0, 4) = 0.0;
  phi_mat(0, 5) = 2 * point.z();
  phi_mat(0, 6) = -2 * point.y();
  phi_mat(0, 7) = 2 * point.y();
  phi_mat(0, 8) = 2 * point.z();
  phi_mat(0, 9) = 0.0;

  // Row 1.
  phi_mat(1, 0) = point.y();
  phi_mat(1, 1) = -point.y();
  phi_mat(1, 2) = point.y();
  phi_mat(1, 3) = -point.y();
  phi_mat(1, 4) = -2.0 * point.z();
  phi_mat(1, 5) = 0.0;
  phi_mat(1, 6) = 2 * point.x();
  phi_mat(1, 7) = 2 * point.x();
  phi_mat(1, 8) = 0.0;
  phi_mat(1, 9) = 2 * point.z();

  // Row 3.
  phi_mat(2, 0) = point.z();
  phi_mat(2, 1) = -point.z();
  phi_mat(2, 2) = -point.z();
  phi_mat(2, 3) = point.z();
  phi_mat(2, 4) = 2.0 * point.y();
  phi_mat(2, 5) = -2.0 * point.x();
  phi_mat(2, 6) = 0.0;
  phi_mat(2, 7) = 0.0;
  phi_mat(2, 8) = 2.0 * point.x();
  phi_mat(2, 9) = 2.0 * point.y();
  return phi_mat;
}

// Check that the input to gDLS* is valid. It checks the inputs have the same
// sizes and that the prior scales are valid.
inline void IsInputDatumValid(const Input& input) {
  CHECK_EQ(input.ray_origins.size(), input.ray_directions.size());
  CHECK_EQ(input.ray_origins.size(), input.world_points.size());
  CHECK_GE(input.priors.scale_penalty, 0.0);
  CHECK_GE(input.priors.gravity_penalty, 0.0);
}

Eigen::Matrix4d ComputeHMatrix(
    const std::vector<Eigen::Vector3d>& ray_origins,
    const std::vector<Eigen::Vector3d>& ray_directions,
    const double scale_penalty_factor) {
  Eigen::Matrix4d h_inverse = Eigen::Matrix4d::Zero();
  for (size_t i = 0; i < ray_directions.size(); ++i) {
    // Computing the scalar in the top-left corner of matrix H^-1.
    h_inverse(0, 0) +=
        ray_origins[i].squaredNorm() -
        ray_origins[i].dot(ray_directions[i]) *
        ray_origins[i].dot(ray_directions[i]);

    // Computing the 3x1 row-vector of the top-right part of matrix H^-1.
    const Eigen::Vector3d temp_term =
        ray_origins[i].dot(ray_directions[i]) * ray_directions[i] -
        ray_origins[i];
    h_inverse.block<3, 1>(1, 0) += temp_term;

    // Computing the 1x3 vector of the bottom-left part of matrix H^-1.
    h_inverse.block<1, 3>(0, 1) += temp_term.transpose();

    // Bottom right 3x3 block of matrix H^-1.
    h_inverse.block<3, 3>(1, 1) +=
        Eigen::Matrix3d::Identity() -
        (ray_directions[i] * ray_directions[i].transpose());
  }

  // Add the scale penalty term to the first entry of h_inverse.
  h_inverse(0, 0) += scale_penalty_factor;

  const Eigen::Matrix4d h_matrix = h_inverse.inverse();

  return h_matrix;
}

// Compute the the matrix F in Eq. xiv i the supplemental material of gDLS*.
// This matrix is F = -B * H.
Eigen::MatrixXd ComputeFMat(
    const std::vector<Eigen::Vector3d>& ray_origins,
    const std::vector<Eigen::Vector3d>& ray_directions,
    const Eigen::Matrix4d& h_matrix) {
  const int num_points = ray_origins.size();
  Eigen::MatrixXd f_mat(num_points, 4);
  Eigen::RowVector4d helper_vec;
  for (size_t i = 0; i < num_points; ++i) {
    // Scalar term: -r_i' c_i.
    helper_vec[0] = -ray_directions[i].dot(ray_origins[i]);
    // Vector term: r'.
    helper_vec.tail(3) = ray_directions[i].transpose();
    // Computing i-th row of matrix F.
    f_mat.block(i, 0, 1, 4) = helper_vec * h_matrix;
  }
  return f_mat;
}

void ComputeScaleAndTranslationFactors(const Input& input,
                                       const Eigen::Matrix4d& h_matrix,
                                       RowVector10d* scale_factor,
                                       Matrix3x10d* translation_factor) {
  // Useful aliases.
  const std::vector<Eigen::Vector3d>& ray_origins = input.ray_origins;
  const std::vector<Eigen::Vector3d>& ray_directions = input.ray_directions;
  const std::vector<Eigen::Vector3d>& world_points = input.world_points;
  const size_t num_correspondences = ray_directions.size();
  Eigen::Matrix<double, 4, 10> sv_helper = Eigen::Matrix<double, 4, 10>::Zero();
  for (size_t i = 0; i < num_correspondences; ++i) {
    const Matrix3x10d left_multiply_matrix = LeftMultiply(world_points[i]);
    // Scale factor.
    sv_helper.row(0) +=
        (ray_origins[i].transpose() - ray_origins[i].dot(ray_directions[i]) *
         ray_directions[i].transpose()) * left_multiply_matrix;

    // Translation factor.
    sv_helper.block<3, 10>(1, 0) +=
        (ray_directions[i] * ray_directions[i].transpose() -
         Eigen::Matrix3d::Identity()) * left_multiply_matrix;
  }

  sv_helper = h_matrix * sv_helper;
  *scale_factor = sv_helper.row(0);
  *translation_factor = sv_helper.block<3, 10>(1, 0);
}

// Computes vector k_i from the scale-constrained gDLS.
inline Eigen::Vector3d ComputeLinearTermPerPoint(
    const int point_idx,
    const Eigen::Matrix4d& h_matrix,
    const Eigen::MatrixXd& f_matrix,
    const Eigen::Vector3d& ray_origin,
    const Eigen::Vector3d& ray_direction,
    const Priors& priors) {
  // Scale linear term.
  const Eigen::Vector3d scalar_linear_term =
      f_matrix(point_idx, 0) * ray_direction - h_matrix.col(0).tail(3) +
      h_matrix(0, 0) * ray_origin;

  // Compute total linear term.
  const Eigen::Vector3d linear_term =
      priors.scale_prior * priors.scale_penalty * scalar_linear_term;
  return linear_term;
}

Matrix10d
ComputeQuadraticPenaltyMatrixFromGravityRegularizer(const Input& input,
                                                    double penalty) {
  // Compute the left-multiply matrix for gravity index.
  const Matrix3x10d world_gravity_dir_matrix =
      LeftMultiply(input.priors.world_down_direction.normalized());
  const Eigen::Vector3d query_gravity_dir =
      input.priors.query_down_direction.normalized();
  const Eigen::Matrix3d query_down_dir_mat =
      ComputeSkewSymmetricMatrix(query_gravity_dir);
  // M = penalty * L(g_I)' * Cross(g_Q)^T * Cross(g_Q) L(g_I),
  // L() is the left-multiply function, and Cross() is the skew-symmetric
  // matrix for cross product. Here g indicates gravity direction according
  // to Q (query) and I (reference).
  const Matrix10d penalty_matrix =
      penalty * world_gravity_dir_matrix.transpose() *
      query_down_dir_mat.transpose() *
      query_down_dir_mat * world_gravity_dir_matrix;
  return penalty_matrix;
}

inline Matrix10d ComputeQuadraticPenaltiesFromScaleConstraint(
    const double scale_penalty,
    const RowVector10d& scale_factor) {
  return scale_penalty * scale_factor.transpose() * scale_factor;
}

inline Vector10d ComputeLinearPenaltiesFromScaleConstraint(
    const Priors& priors,
    const Eigen::Matrix4d& h_matrix,
    const RowVector10d& scale_factor) {
  const double& scale_prior = priors.scale_prior;
  const double& scale_penalty = priors.scale_penalty;
  const double scalar =
      scale_penalty * scale_prior * h_matrix(0, 0) - scale_prior;
  return scale_penalty * scalar * scale_factor;
}

inline double ComputeConstantTermFromScaleConstraint(
    const Priors& priors,
    const Eigen::Matrix4d& h_matrix) {
  const double& scale_penalty = priors.scale_penalty;
  const double& scale_prior = priors.scale_prior;
  const double scale_term =
      scale_penalty * scale_prior * h_matrix(0, 0) - scale_prior;
  const double term = scale_term * scale_term;
  return scale_penalty * term;
}

// The observed pattern is that duplicate rotations appear consequtively in the
// vector, i.e., rotation[i] == rotation[i + 1] is common.
std::vector<Eigen::Quaterniond> RemoveDuplicateRotations(
    const std::vector<Eigen::Quaterniond>& candidate_rotations) {
  const double kAngleThreshold = DegToRad(0.1);
  std::vector<Eigen::Quaterniond> rotations;
  rotations.reserve(candidate_rotations.size());

  // If no rotations then return empty vector.
  if (candidate_rotations.empty()) {
    return rotations;
  }

  for (int i = 0; i < candidate_rotations.size(); ++i) {
    bool duplicate_rotation = false;
    const Eigen::Quaterniond& candidate_rotation = candidate_rotations[i];
    for (int j = rotations.size() - 1; j >= 0; --j) {
      if (candidate_rotation.angularDistance(rotations[j]) < kAngleThreshold) {
        duplicate_rotation = true;
        break;
      }
    }
    if (!duplicate_rotation) {
      rotations.push_back(candidate_rotation);
    }
  }
  return rotations;
}

// Constructs the vector s as indicated in Eq. 13 of gDLS* main paper.
inline Vector10d ComputeRotationVector(const Eigen::Quaterniond& rotation) {
  Vector10d rotation_vector;
  // Set the values of the rotation vector.
  rotation_vector[0] = rotation.w() * rotation.w();
  rotation_vector[1] = rotation.x() * rotation.x();
  rotation_vector[2] = rotation.y() * rotation.y();
  rotation_vector[3] = rotation.z() * rotation.z();
  rotation_vector[4] = rotation.w() * rotation.x();
  rotation_vector[5] = rotation.w() * rotation.y();
  rotation_vector[6] = rotation.w() * rotation.z();
  rotation_vector[7] = rotation.x() * rotation.y();
  rotation_vector[8] = rotation.x() * rotation.z();
  rotation_vector[9] = rotation.y() * rotation.z();
  return rotation_vector;
}

std::vector<Eigen::Vector3d> ComputeScalesAndTranslations(
    const std::vector<Eigen::Quaterniond>& rotations,
    const Matrix3x10d& translation_factor,
    const Input& input,
    const Eigen::Matrix4d& h_matrix,
    const RowVector10d& scale_factor,
    std::vector<double>* scales) {
  // Solve for translation as a function of rotation.
  std::vector<Eigen::Vector3d> translations;
  translations.reserve(rotations.size());
  CHECK_NOTNULL(scales)->reserve(rotations.size());
  const double& scale_penalty = input.priors.scale_penalty;
  for (const Eigen::Quaterniond& rotation : rotations) {
    const Vector10d rotationVector = ComputeRotationVector(rotation);
    translations.emplace_back(
        translation_factor * rotationVector +
        scale_penalty * input.priors.scale_prior * h_matrix.col(0).tail(3));
    scales->emplace_back(
        scale_factor * rotationVector +
        scale_penalty * input.priors.scale_prior * h_matrix(0, 0));
  }
  return translations;
}

void DiscardBadSolutions(const Input& input_datum, Solution* solution) {
  // Useful aliases.
  std::vector<Eigen::Quaterniond>& solution_rotations = solution->rotations;
  std::vector<Eigen::Vector3d>& solution_translations = solution->translations;
  std::vector<double>& solution_scales = solution->scales;
  CHECK_EQ(solution_rotations.size(), solution_translations.size());
  CHECK_EQ(solution_rotations.size(), solution_scales.size());
  std::vector<Eigen::Quaterniond> final_rotations;
  std::vector<Eigen::Vector3d> final_translations;
  std::vector<double> final_scales;
  final_rotations.reserve(solution_rotations.size());
  final_translations.reserve(solution_translations.size());
  final_scales.reserve(solution_scales.size());

  const std::vector<Eigen::Vector3d>& world_points = input_datum.world_points;
  const std::vector<Eigen::Vector3d>& ray_origins = input_datum.ray_origins;
  const std::vector<Eigen::Vector3d>& ray_directions =
      input_datum.ray_directions;

  // For every computed solution, check that points are in front of camera.
  for (int i = 0; i < solution_rotations.size(); ++i) {
    const Eigen::Quaterniond& soln_rotation = solution_rotations[i];
    const Eigen::Vector3d& soln_translation = solution_translations[i];
    const double scale = solution_scales[i];

    // Check that all points are in front of the camera. Discard the solution
    // if this is not the case.
    bool all_points_in_front_of_camera = true;

    for (int j = 0; j < world_points.size(); ++j) {
      const Eigen::Vector3d transformed_point =
          soln_rotation * world_points[j] + soln_translation -
          scale * ray_origins[j];

      // Find the rotation that puts the image ray at unit Z direction.
      const Eigen::Quaterniond unrot =
          Eigen::Quaterniond::FromTwoVectors(ray_directions[j],
                                             Eigen::Vector3d::UnitZ());

      // Rotate the transformed point and check if the z coordinate is
      // negative. This will indicate if the point is projected behind the
      // camera.
      const Eigen::Vector3d rotated_projection = unrot * transformed_point;
      if (rotated_projection.z() < 0) {
        all_points_in_front_of_camera = false;
        break;
      }
    }

    if (all_points_in_front_of_camera) {
      final_rotations.emplace_back(soln_rotation);
      final_translations.emplace_back(soln_translation);
      final_scales.push_back(scale);
    }
  }

  // Set the final solutions.
  std::swap(solution_rotations, final_rotations);
  std::swap(solution_translations, final_translations);
  std::swap(solution_scales, final_scales);
}

}  // namespace

void GdlsStar::ComputeHelperMatrices(const Input& input) {
  // Compute helper matrices.
  // Compute H matrix from.
  helper_matrices_.h_matrix =
      ComputeHMatrix(input.ray_origins,
                     input.ray_directions,
                     input.priors.scale_penalty);

  // Compute the F matrix, they are needed for the translation prior part.
  helper_matrices_.f_matrix = ComputeFMat(input.ray_origins,
                                          input.ray_directions,
                                          helper_matrices_.h_matrix);

  // Compute scale and translation factors.
  ComputeScaleAndTranslationFactors(input,
                                    helper_matrices_.h_matrix,
                                    &helper_matrices_.scale_factor,
                                    &helper_matrices_.translation_factor);
}

void GdlsStar::ComputeLeastSquaresCostParameters(const Input& input) {
  // Useful aliases.
  const std::vector<Eigen::Vector3d>& ray_origins = input.ray_origins;
  const std::vector<Eigen::Vector3d>& ray_directions = input.ray_directions;
  const std::vector<Eigen::Vector3d>& world_points = input.world_points;
  const size_t num_correspondences = ray_directions.size();

  cost_params_.quadratic_penalty_mat.setZero();
  cost_params_.linear_penalty_vector.setZero();
  cost_params_.gamma = 0.0;
  for (int i = 0; i < num_correspondences; ++i) {
    // Compute the quadratic term per point.
    const Matrix3x10d left_multiply_matrix = LeftMultiply(world_points[i]);
    const Matrix3x10d cost_coeff_term =
        (ray_directions[i] * ray_directions[i].transpose() -
         Eigen::Matrix3d::Identity()) *
        (LeftMultiply(world_points[i]) -
         ray_origins[i] * helper_matrices_.scale_factor +
         helper_matrices_.translation_factor);
    cost_params_.quadratic_penalty_mat +=
        cost_coeff_term.transpose() * cost_coeff_term;

    // Compute the linear term per point as calculated in Eq. 22 of scale and
    // gravity constrained gdls.
    const Eigen::Vector3d scalar_penalty_vector =
        ComputeLinearTermPerPoint(i,
                                  helper_matrices_.h_matrix,
                                  helper_matrices_.f_matrix,
                                  ray_origins[i],
                                  ray_directions[i],
                                  input.priors);
    cost_params_.linear_penalty_vector +=
        scalar_penalty_vector.transpose() * cost_coeff_term;

    // Compute the constant term per point.
    cost_params_.gamma += scalar_penalty_vector.dot(scalar_penalty_vector);
  }

  // Add penalties for gravity constraints.
  if (input.priors.gravity_penalty > 0.0) {
    cost_params_.quadratic_penalty_mat +=
        ComputeQuadraticPenaltyMatrixFromGravityRegularizer(
            input, input.priors.gravity_penalty);
  }

  // Add penalties for scale constraints.
  if (input.priors.scale_penalty > 0.0) {
    cost_params_.quadratic_penalty_mat +=
        ComputeQuadraticPenaltiesFromScaleConstraint(
            input.priors.scale_penalty,
            helper_matrices_.scale_factor);
    cost_params_.linear_penalty_vector +=
        ComputeLinearPenaltiesFromScaleConstraint(
            input.priors,
            helper_matrices_.h_matrix,
            helper_matrices_.scale_factor);
    cost_params_.gamma +=
        ComputeConstantTermFromScaleConstraint(
            input.priors,
            helper_matrices_.h_matrix);
  }
}

CostParameters GdlsStar::ComputeCostParameters(const Input& input) {
  // Validate input.
  IsInputDatumValid(input);

  // Compute all the helper matrices.
  ComputeHelperMatrices(input);

  // Compute the least squares cost parameters.
  ComputeLeastSquaresCostParameters(input);

  return cost_params_;
}

std::vector<Eigen::Quaterniond> GdlsStar::EstimateRotations() {
  std::vector<Eigen::Quaterniond> rotations(kNumMaxRotationsExploitingSymmetry);
  // Build action matrix.
  const Matrix8d action_matrix = theia::BuildActionMatrixUsingSymmetry(
      cost_params_.quadratic_penalty_mat,
      cost_params_.linear_penalty_vector,
      &template_matrix_);

  const Eigen::EigenSolver<Matrix8d> eigen_solver(action_matrix);
  const Matrix8cd eigen_vectors = eigen_solver.eigenvectors();

  for (int i = 0; i < rotations.size(); ++i) {
    // Complex solutions can be good, in particular when the number of
    // correspondences is really low. To use these complex solutions, we simply
    // ignore their imaginary part.
    rotations[i] = Eigen::Quaterniond(eigen_vectors(4, i).real(),
                                      eigen_vectors(5, i).real(),
                                      eigen_vectors(6, i).real(),
                                      eigen_vectors(7, i).real()).normalized();
  }

  return RemoveDuplicateRotations(rotations);
}

bool GdlsStar::EstimateSimilarityTransformation(const Input& input,
                                                Solution* solution) {
  CHECK_NOTNULL(solution)->rotations.clear();
  solution->translations.clear();
  solution->scales.clear();

  // Construct cost parameters.
  ComputeCostParameters(input);

  // Estimate rotations.
  solution->rotations = EstimateRotations();

  // Compute translations and scales.
  solution->translations =
      ComputeScalesAndTranslations(solution->rotations,
                                   helper_matrices_.translation_factor,
                                   input,
                                   helper_matrices_.h_matrix,
                                   helper_matrices_.scale_factor,
                                   &solution->scales);

  // Discard solutions that do not have the points in front of the camera.
  DiscardBadSolutions(input, solution);
  return !solution->rotations.empty();
}

}  // namespace msft

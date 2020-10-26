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

#include "gtest/gtest.h"

#include <array>
#include <random>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include "gdls_star/gdls_star.h"

namespace msft {
namespace {
using Eigen::AngleAxisd;
using Eigen::Quaterniond;
using Eigen::Vector3d;

using Input = GdlsStar::Input;

// Convert degrees to radians.
constexpr double DegToRad(double angle_degrees) {
  constexpr double kDegToRad = M_PI / 180.0;
  return angle_degrees * kDegToRad;
}

// Convert radiants to degrees.
constexpr double RadToDeg(double angle_radians) {
  constexpr double kRadToDeg = 180.0 / M_PI;
  return angle_radians * kRadToDeg;
}

struct TestParams {
  double max_reprojection_error;
  double max_rotation_difference;
  double max_translation_difference;
  double max_scale_difference;
  double projection_noise_std_dev;
  double down_direction_noise_std_dev;
};

struct ExpectedSolution {
  Quaterniond rotation;
  Vector3d translation;
  double scale;
};

struct TestInput {
  std::vector<Vector3d> camera_centers;
  std::vector<Vector3d> world_points;
  Vector3d query_down_direction;
  Vector3d world_down_direction;
  double scale_prior;
  double scale_penalty;
  double gravity_penalty;
};

class GdlsStarTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    estimator = new GdlsStar;
    random_engine = new std::default_random_engine;
    prng = new std::mt19937(std::mt19937::default_seed);
    gaussian_distribution = new std::normal_distribution<double>(0.0, 1.0);
  }

  static void TearDownTestCase() {
    delete prng;
    delete random_engine;
    delete estimator;
    delete gaussian_distribution;
  }

  static Input GenerateTestingData(const ExpectedSolution& expected_solution,
                                   const TestParams& test_params,
                                   const TestInput& test_input);

  static void TestWithNoise(const ExpectedSolution& expected_solution,
                            const TestParams& test_params,
                            const TestInput& test_input);

  static void AddNoiseToRay(const double std_dev, Eigen::Vector3d* ray);

  // Estimator.
  static GdlsStar* estimator;
  static std::default_random_engine* random_engine;
  static std::mt19937* prng;
  static std::normal_distribution<double>* gaussian_distribution;
};

GdlsStar* GdlsStarTest::estimator = nullptr;
std::default_random_engine* GdlsStarTest::random_engine = nullptr;
std::mt19937* GdlsStarTest::prng = nullptr;
std::normal_distribution<double>* GdlsStarTest::gaussian_distribution = nullptr;

void GdlsStarTest::AddNoiseToRay(const double std_dev, Eigen::Vector3d* ray) {
  const double scale = CHECK_NOTNULL(ray)->norm();
  const Eigen::Quaterniond rotation =
      Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(0, 0, 1.0), *ray);
  const Eigen::Vector3d noisy_point(std_dev * (*gaussian_distribution)(*prng),
                                    std_dev * (*gaussian_distribution)(*prng),
                                    1);
  *ray = (rotation * (scale * noisy_point)).normalized();
}

Input GdlsStarTest::GenerateTestingData(
    const ExpectedSolution& expected_solution,
    const TestParams& test_params,
    const TestInput& test_input) {
  Input input;
  const int num_points = test_input.world_points.size();
  const int num_cameras = test_input.camera_centers.size();
  std::vector<Vector3d>& camera_rays = input.ray_directions;
  std::vector<Vector3d>& ray_origins = input.ray_origins;
  camera_rays.reserve(num_points);
  ray_origins.reserve(num_points);
  input.world_points = test_input.world_points;
  const std::vector<Vector3d>& world_points = test_input.world_points;
  const std::vector<Vector3d>& camera_centers = test_input.camera_centers;
  const Quaterniond expected_rotation = expected_solution.rotation;
  const Vector3d expected_translation = expected_solution.translation;
  const double expected_scale = expected_solution.scale;
  for (int i = 0; i < num_points; ++i) {
    // Setting camera positions or ray origins.
    ray_origins.emplace_back(
        (expected_rotation * camera_centers[i % num_cameras] +
         expected_translation) / expected_scale);

    // Reproject 3D points into camera frame.
    camera_rays.emplace_back(
        (expected_rotation * world_points[i] + expected_translation -
         expected_scale * ray_origins[i]).normalized());
  }

  // Add noise to the camera rays.
  if (test_params.projection_noise_std_dev > 0.0) {
    // Adds noise to both of the rays.
    for (int i = 0; i < num_points; ++i) {
      AddNoiseToRay(test_params.projection_noise_std_dev, &camera_rays[i]);
    }
  }

  // Compute the query down direction.
  input.priors.query_down_direction =
      expected_rotation * input.priors.world_down_direction;

  if (test_params.down_direction_noise_std_dev > 0.0) {
    AddNoiseToRay(test_params.down_direction_noise_std_dev,
                  &input.priors.world_down_direction);
    AddNoiseToRay(test_params.down_direction_noise_std_dev,
                  &input.priors.query_down_direction);
  }

  // Setting penalties for the regularizers.
  input.priors.scale_penalty = test_input.scale_penalty;
  input.priors.gravity_penalty = test_input.gravity_penalty;
  input.priors.scale_prior = test_input.scale_prior;
  return input;
}

bool CheckReprojectionErrors(const Input& input,
                             const Eigen::Quaterniond& soln_rotation,
                             const Eigen::Vector3d& soln_translation,
                             const double soln_scale,
                             const double max_reprojection_error) {
  const int num_points = input.world_points.size();
  const std::vector<Vector3d>& camera_rays = input.ray_directions;
  const std::vector<Vector3d>& ray_origins = input.ray_origins;
  const std::vector<Vector3d>& world_points = input.world_points;
  double good_reprojection_errors = true;
  for (int i = 0; i < num_points; ++i) {
    const Quaterniond unrot =
        Quaterniond::FromTwoVectors(camera_rays[i], Vector3d(0, 0, 1));
    const Vector3d reprojected_point =
        (soln_rotation * world_points[i] + soln_translation) /
        soln_scale - ray_origins[i];

    const Vector3d unrot_cam_ray = unrot * camera_rays[i];
    const Vector3d unrot_reproj_pt = unrot * reprojected_point;
    
    const double reprojection_error = 
        (unrot_cam_ray.hnormalized() - unrot_reproj_pt.hnormalized()).norm();
    good_reprojection_errors = (good_reprojection_errors &&
                                (reprojection_error < max_reprojection_error));
  }
  return good_reprojection_errors;
}

void GdlsStarTest::TestWithNoise(
    const ExpectedSolution& expected_solution,
    const TestParams& test_params,
    const TestInput& test_input) {
  // Generate testing data.
  const Input input =
      GenerateTestingData(expected_solution, test_params, test_input);

  // Estimate.
  GdlsStar::Solution solution;
  EXPECT_TRUE(estimator->EstimateSimilarityTransformation(input, &solution));

  // Check the solutions here.
  bool matched_transform = false;
  EXPECT_GT(solution.rotations.size(), 0);
  for (int i = 0; i < solution.rotations.size(); ++i) {
    const bool good_reprojection_errors =
        CheckReprojectionErrors(input,
                                solution.rotations[i],
                                solution.translations[i],
                                solution.scales[i],
                                test_params.max_reprojection_error);
    const double rotation_difference =
        expected_solution.rotation.angularDistance(solution.rotations[i]);
    const bool matched_rotation =
        rotation_difference < test_params.max_rotation_difference;
    const double translation_difference =
        (expected_solution.translation -
         solution.translations[i]).squaredNorm();
    const bool matched_translation =
        translation_difference < test_params.max_translation_difference;
    const double scale_difference =
        fabs(expected_solution.scale - solution.scales[i]);
    const bool matched_scale =
        scale_difference < test_params.max_scale_difference;
    VLOG(3) << "Matched rotation: " << matched_rotation
            << " rotation error [deg]=" << RadToDeg(rotation_difference);
    VLOG(3) << "Matched translation: " << matched_translation
            << " translation error=" << translation_difference;
    VLOG(3) << "Matched scale: " << matched_scale
            << " scale difference=" << scale_difference;
    VLOG(3) << "Good reprojection errors: " << good_reprojection_errors;
    if (matched_rotation && matched_translation && matched_scale &&
        good_reprojection_errors) {
      matched_transform = true;
    }
  }
  EXPECT_TRUE(matched_transform);
}

TEST_F(GdlsStarTest, UnconstrainedBasicTest) {
  const std::vector<Vector3d> kPoints3d = { Vector3d(-1.0, 3.0, 3.0),
                                            Vector3d(1.0, -1.0, 2.0),
                                            Vector3d(-1.0, 1.0, 2.0),
                                            Vector3d(2.0, 1.0, 3.0) };
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };
  // Expected solution.
  ExpectedSolution solution;
  solution.rotation =
      Quaterniond(AngleAxisd(DegToRad(13.0), Vector3d(0.0, 0.0, 1.0)));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.5;

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 0.0;
  test_params.down_direction_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-2;

  // Test input.
  TestInput test_input;
  test_input.camera_centers = kImageOrigins;
  test_input.world_points = kPoints3d;
  test_input.query_down_direction.setZero();
  test_input.world_down_direction.setZero();
  test_input.scale_penalty = 0.0;
  test_input.gravity_penalty = 0.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(GdlsStarTest, UnconstrainedBasicTestWithNoise) {
  const std::vector<Vector3d> kPoints3d = { Vector3d(-1.0, 3.0, 3.0),
                                            Vector3d(1.0, -1.0, 2.0),
                                            Vector3d(-1.0, 1.0, 2.0),
                                            Vector3d(2.0, 1.0, 3.0) };
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };
  // Expected solution.
  ExpectedSolution solution;
  solution.rotation =
      Quaterniond(AngleAxisd(DegToRad(13.0), Vector3d(0.0, 0.0, 1.0)));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.5;

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 1.0 / 512.0;
  test_params.down_direction_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-2;

  // Test input.
  TestInput test_input;
  test_input.camera_centers = kImageOrigins;
  test_input.world_points = kPoints3d;
  test_input.query_down_direction.setZero();
  test_input.world_down_direction.setZero();
  test_input.scale_penalty = 0.0;
  test_input.gravity_penalty = 0.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(GdlsStarTest, GravityRegularizedBasicTest) {
  const std::vector<Vector3d> kPoints3d = { Vector3d(-1.0, 3.0, 3.0),
                                            Vector3d(1.0, -1.0, 2.0),
                                            Vector3d(-1.0, 1.0, 2.0),
                                            Vector3d(2.0, 1.0, 3.0) };
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };
  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(
      AngleAxisd(DegToRad(13.0), Vector3d(1.0, 1.0, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.5;

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 0.0;
  test_params.down_direction_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-2;

  // Test input.
  TestInput test_input;
  test_input.camera_centers = kImageOrigins;
  test_input.world_points = kPoints3d;
  test_input.query_down_direction.setZero();
  test_input.world_down_direction = Vector3d::UnitZ();
  test_input.scale_penalty = 0.0;
  test_input.gravity_penalty = 1.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(GdlsStarTest, GravityRegularizedBasicTestWithNoise) {
  const std::vector<Vector3d> kPoints3d = { Vector3d(-1.0, 3.0, 3.0),
                                            Vector3d(1.0, -1.0, 2.0),
                                            Vector3d(-1.0, 1.0, 2.0),
                                            Vector3d(2.0, 1.0, 3.0) };
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };
  // Expected solution.
  ExpectedSolution solution;
  solution.rotation =
      Quaterniond(
          AngleAxisd(DegToRad(13.0), Vector3d(1.0, 1.0, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.5;

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 1.0 / 512.0;
  test_params.down_direction_noise_std_dev = 1.0 / 512.0;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-2;
  test_params.max_scale_difference = 1e-2;

  // Test input.
  TestInput test_input;
  test_input.camera_centers = kImageOrigins;
  test_input.world_points = kPoints3d;
  test_input.query_down_direction.setZero();
  test_input.world_down_direction = Vector3d::UnitZ();
  test_input.scale_penalty = 0.0;
  test_input.gravity_penalty = 1.0;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(GdlsStarTest, ScaleRegularizedBasicTest) {
  const std::vector<Vector3d> kPoints3d = { Vector3d(-1.0, 3.0, 3.0),
                                            Vector3d(1.0, -1.0, 2.0),
                                            Vector3d(-1.0, 1.0, 2.0),
                                            Vector3d(2.0, 1.0, 3.0) };
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };
  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(
      AngleAxisd(DegToRad(13.0), Vector3d(1.0, 0.5, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.5;

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 0.0;
  test_params.down_direction_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-2;

  // Test input.
  TestInput test_input;
  test_input.camera_centers = kImageOrigins;
  test_input.world_points = kPoints3d;
  test_input.query_down_direction.setZero();
  test_input.world_down_direction.setZero();
  test_input.scale_penalty = 1.0;
  test_input.gravity_penalty = 0.0;
  test_input.scale_prior = solution.scale;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(GdlsStarTest, ScaleRegularizedBasicTestWithNoise) {
  const std::vector<Vector3d> kPoints3d = { Vector3d(-1.0, 3.0, 3.0),
                                            Vector3d(1.0, -1.0, 2.0),
                                            Vector3d(-1.0, 1.0, 2.0),
                                            Vector3d(2.0, 1.0, 3.0) };
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };
  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(
      AngleAxisd(DegToRad(13.0), Vector3d(1.0, 0.5, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.5;

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 1.0 / 512;
  test_params.down_direction_noise_std_dev = 0.0;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-2;
  test_params.max_scale_difference = 1e-2;

  // Test input.
  TestInput test_input;
  test_input.camera_centers = kImageOrigins;
  test_input.world_points = kPoints3d;
  test_input.query_down_direction.setZero();
  test_input.world_down_direction.setZero();
  test_input.scale_penalty = 1.0;
  test_input.gravity_penalty = 0.0;
  test_input.scale_prior = solution.scale;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(GdlsStarTest, ScaleAndGravityRegularizedBasicTestWithNoise) {
  const std::vector<Vector3d> kPoints3d = { Vector3d(-1.0, 3.0, 3.0),
                                            Vector3d(1.0, -1.0, 2.0),
                                            Vector3d(-1.0, 1.0, 2.0),
                                            Vector3d(2.0, 1.0, 3.0) };
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };
  // Expected solution.
  ExpectedSolution solution;
  solution.rotation = Quaterniond(
      AngleAxisd(DegToRad(13.0), Vector3d(1.0, 0.5, 1.0).normalized()));
  solution.translation = Vector3d(1.0, 1.0, 1.0);
  solution.scale = 2.5;

  // Test parameters.
  TestParams test_params;
  test_params.projection_noise_std_dev = 1.0 / 512;
  test_params.down_direction_noise_std_dev = 1.0 / 512;
  test_params.max_reprojection_error =  5.0 / 512;
  test_params.max_rotation_difference = DegToRad(1.0);
  test_params.max_translation_difference = 1e-3;
  test_params.max_scale_difference = 1e-2;

  // Test input.
  TestInput test_input;
  test_input.camera_centers = kImageOrigins;
  test_input.world_points = kPoints3d;
  test_input.query_down_direction.setZero();
  test_input.world_down_direction = Vector3d::UnitZ();
  test_input.scale_penalty = 1.0;
  test_input.gravity_penalty = 1.0;
  test_input.scale_prior = solution.scale;

  TestWithNoise(solution, test_params, test_input);
}

TEST_F(GdlsStarTest, ScaleAndGravityRegularizedTestWithNoiseAndManyPoints) {
  // Sets some test rotations and translations.
  constexpr int kNumAxes = 8;
  constexpr int kNumPointSets = 3;
  const std::vector<Vector3d> kImageOrigins = { Vector3d(-1.0, 0.0, 0.0),
                                                Vector3d(0.0, 0.0, 0.0),
                                                Vector3d(2.0, 0.0, 0.0),
                                                Vector3d(3.0, 0.0, 0.0) };

  const std::array<int, kNumPointSets> num_points = { 100, 500, 1000 };
  std::uniform_real_distribution angle_distribution(0.0, RadToDeg(33));
  std::uniform_real_distribution scale_distribution(0.0, RadToDeg(50));
  std::uniform_real_distribution point_distribution(0.0, 1.0);
  std::vector<Vector3d> points_3d;

  int test_trial = 0;
  for (int i = 0; i < kNumAxes; ++i) {
    const Quaterniond soln_rotation(
        AngleAxisd(angle_distribution(*prng), Vector3d::Random().normalized()));
    for (int j = 0; j < kNumPointSets; ++j) {
      points_3d.clear();
      points_3d.reserve(num_points[j]);
      for (int k = 0; k < num_points[j]; ++k) {
        points_3d.emplace_back(10 * point_distribution(*prng) - 5,
                               10 * point_distribution(*prng) - 5,
                               8 * point_distribution(*prng) + 2);
      }
  
      // Expected solution.
      ExpectedSolution solution;
      solution.rotation = soln_rotation;
      solution.translation = Vector3d::Random();
      solution.scale = scale_distribution(*prng);

      // Test parameters.
      TestParams test_params;
      test_params.projection_noise_std_dev = 1.0 / 512;
      test_params.down_direction_noise_std_dev = 1.0 / 512;
      test_params.max_reprojection_error =  8.0 / 512;
      test_params.max_rotation_difference = DegToRad(2.0);
      test_params.max_translation_difference = 1e-2;
      test_params.max_scale_difference = 0.1;

      // Test input.
      TestInput test_input;
      test_input.camera_centers = kImageOrigins;
      test_input.world_points = std::move(points_3d);
      test_input.query_down_direction.setZero();
      test_input.world_down_direction = Vector3d::UnitZ();
      test_input.scale_penalty = 1.0;
      test_input.gravity_penalty = 1.0;
      test_input.scale_prior = solution.scale;

      VLOG(3) << ">>>> Test trial: " << ++test_trial;
      TestWithNoise(solution, test_params, test_input);
    }
  }

}

}  // namespace
}  // namespace msft

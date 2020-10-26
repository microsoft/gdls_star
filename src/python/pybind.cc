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

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "gdls_star/pinhole_camera.h"
#include "gdls_star/camera_feature_correspondence_2d_3d.h"
#include "gdls_star/estimate_similarity_transformation.h"
#include "gdls_star/gdls_star.h"
#include "gdls_star/gdls_star_robust_estimator.h"
#include "gdls_star/util.h"

namespace py = pybind11;

using msft::PinholeCamera;
using msft::CameraFeatureCorrespondence2D3D;
using msft::GdlsStar;
using msft::GdlsStarRobustEstimator;

struct PySolution {
  Eigen::Vector4d rotation = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  double scale = 1.0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct PyRansacSolution {
  PySolution best_solution;
  std::vector<int> inliers;
};

std::vector<PySolution> EstimateSimilarityTransformation(
    const GdlsStar::Priors& priors,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences) {
  std::vector<PySolution> final_solutions;
  GdlsStar::Solution solution;
  // Compute input datum.
  const GdlsStar::Input input = msft::ComputeInputDatum(correspondences);
  GdlsStar estimator;
  estimator.EstimateSimilarityTransformation(input, &solution);
  final_solutions.resize(solution.rotations.size());
  for (int i = 0; i < solution.rotations.size(); ++i) {
    const Eigen::Quaterniond& rotation = solution.rotations[i];
    final_solutions[i].rotation = Eigen::Vector4d(rotation.w(),
                                                  rotation.x(),
                                                  rotation.y(),
                                                  rotation.z());
    final_solutions[i].translation = solution.translations[i];
    final_solutions[i].scale = solution.scales[i];
  }
  return final_solutions;
}

// Computes the similarity transformation given 2D-3D correspondences and priors
// using a RANSAC estimator.
PyRansacSolution EstimateSimilarityTransformationViaRansac(
    const GdlsStarRobustEstimator::RansacParameters& params,
    const GdlsStar::Priors& priors,
    const std::vector<CameraFeatureCorrespondence2D3D>& correspondences) {
  PyRansacSolution final_solution;
  GdlsStarRobustEstimator::RansacSummary ransac_summary;
  GdlsStarRobustEstimator estimator(params);
  const GdlsStar::Solution solution =
      estimator.Estimate(priors, correspondences, &ransac_summary);
  const Eigen::Quaterniond& rotation = solution.rotations[0];
  final_solution.best_solution.rotation = Eigen::Vector4d(rotation.w(),
                                                          rotation.x(),
                                                          rotation.y(),
                                                          rotation.z());
  final_solution.best_solution.translation = solution.translations[0];
  final_solution.best_solution.scale = solution.scales[0];
  final_solution.inliers = ransac_summary.inliers;
  return final_solution;
}

PYBIND11_MODULE(pygdls_star, module) {
  module.doc() = "gDLS* Python module"; // Optional module docstring.

  // Ransac parameter class.
  py::class_<GdlsStarRobustEstimator::RansacParameters>(module, "RansacParams")
      .def(py::init<>())
      .def_readwrite("failure_probability",
                     &GdlsStarRobustEstimator::RansacParameters::failure_probability)
      .def_readwrite("reprojection_error_thresh",
                     &GdlsStarRobustEstimator::RansacParameters::reprojection_error_thresh)
      .def_readwrite("min_iterations",
                     &GdlsStarRobustEstimator::RansacParameters::min_iterations)
      .def_readwrite("max_iterations",
                     &GdlsStarRobustEstimator::RansacParameters::max_iterations);

  // Ransac summary class.
  py::class_<GdlsStarRobustEstimator::RansacSummary>(module, "RansacSummary")
      .def(py::init<>())
      .def_readwrite("inliers",
                     &GdlsStarRobustEstimator::RansacSummary::inliers)
      .def_readwrite("num_iterations",
                     &GdlsStarRobustEstimator::RansacSummary::num_iterations)
      .def_readwrite("confidence",
                     &GdlsStarRobustEstimator::RansacSummary::confidence)
      .def_readwrite("num_hypotheses",
                     &GdlsStarRobustEstimator::RansacSummary::num_hypotheses);

  // Pinhole camera.
  py::class_<PinholeCamera>(module, "PinholeCamera")
      .def(py::init<const double,
                    const Eigen::Vector2d,
                    const Eigen::Vector4d,
                    const Eigen::Vector3d>(),
           py::arg("focal_length") = 1.0,
           py::arg("principal_point") = Eigen::Vector2d::Zero(),
           py::arg("world_to_cam_rot") = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0),
           py::arg("world_to_cam_trans") = Eigen::Vector3d::Zero())
      .def("project_point", &PinholeCamera::ProjectPoint)
      .def("pixel_to_unit_ray", &PinholeCamera::PixelToUnitRay)
      .def("get_position", &PinholeCamera::GetPosition);

  // CameraFeatureCorrespondence2D3D.
  py::class_<CameraFeatureCorrespondence2D3D>(module,
                                              "CameraFeatureCorrespondence2D3D")
      .def(py::init<>())
      .def_readwrite("camera", &CameraFeatureCorrespondence2D3D::camera)
      .def_readwrite("observation",
                     &CameraFeatureCorrespondence2D3D::observation)
      .def_readwrite("point", &CameraFeatureCorrespondence2D3D::point);

  // Priors.
  py::class_<GdlsStar::Priors>(module, "Priors")
      .def(py::init<>())
      .def_readwrite("world_down_direction",
                     &GdlsStar::Priors::world_down_direction)
      .def_readwrite("query_down_direction",
                     &GdlsStar::Priors::query_down_direction)
      .def_readwrite("scale_penalty", &GdlsStar::Priors::scale_penalty)
      .def_readwrite("scale_prior", &GdlsStar::Priors::scale_prior)
      .def_readwrite("gravity_penalty", &GdlsStar::Priors::gravity_penalty);

  // PySolution.
  py::class_<PySolution>(module, "Solution")
      .def(py::init<>())
      .def_readwrite("rotation", &PySolution::rotation)
      .def_readwrite("translation", &PySolution::translation)
      .def_readwrite("scale", &PySolution::scale);

  // PyRansacSolution.
  py::class_<PyRansacSolution>(module, "RansacSolution")
      .def(py::init<>())
      .def_readwrite("best_solution", &PyRansacSolution::best_solution)
      .def_readwrite("inliers", &PyRansacSolution::inliers);

  // Estimate similarity transformation using plain gDLS*.
  module.def("estimate", &EstimateSimilarityTransformation);
  module.def("estimate_ransac", &EstimateSimilarityTransformationViaRansac);
}

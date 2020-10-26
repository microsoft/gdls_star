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

#ifndef GDLS_STAR_MATH_UTILS_H_
#define GDLS_STAR_MATH_UTILS_H_
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace msft {

// Convert degrees to radians.
constexpr double DegToRad(double angle_degrees) noexcept {
  constexpr double kDegToRad = M_PI / 180.0;
  return angle_degrees * kDegToRad;
}

}  // namespace msft

#endif  // GDLS_STAR_MATH_UTILS_H_


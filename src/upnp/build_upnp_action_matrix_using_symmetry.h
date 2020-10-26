// This code comes from the Theia-SfM library:
//    https://github.com/sweeneychris/TheiaSfM
//
// Copyright (C) 2018 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Victor Fragoso (victor.fragoso@mail.wvu.edu)

#ifndef THEIA_SFM_POSE_BUILD_UPNP_ACTION_MATRIX_USING_SYMMETRY_H_
#define THEIA_SFM_POSE_BUILD_UPNP_ACTION_MATRIX_USING_SYMMETRY_H_

#include <Eigen/Core>

namespace theia {

using Matrix10d = Eigen::Matrix<double, 10, 10>;
using Vector10d = Eigen::Matrix<double, 10, 1>;
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Builds an action matrix for estimating rotations given Upnp cost parameters.
// The function returns the 8x8 action matrix since exploits symmetry in the
// underlying polynomial system of Upnp.
//
// Params:
//   a_matrix:  The quadratic penalty matrix.
//   b_vector:  The linear penalty vector.
//   template_matrix:  The template matrix buffer.
Eigen::Matrix<double, 8, 8> BuildActionMatrixUsingSymmetry(
    const Matrix10d& a_matrix,
    const Vector10d& b_vector,
    RowMajorMatrixXd* template_matrix);

inline Eigen::Matrix<double, 8, 8> BuildActionMatrixUsingSymmetry(
    const Matrix10d& a_matrix,
    const Vector10d& b_vector) {
  RowMajorMatrixXd template_matrix;
  return BuildActionMatrixUsingSymmetry(a_matrix, b_vector, &template_matrix);
}

}  // namespace theia

#endif  // THEIA_SFM_POSE_BUILD_UPNP_ACTION_MATRIX_USING_SYMMETRY_H_

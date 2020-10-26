# This code comes from Allen:
#  https://gitlab.cern.ch/lhcb/Allen
#
# - Find the NumPy libraries

# This module finds if Pybind11 is installed, and sets the following variables
# indicating where it is.
#
# TODO: Update to provide the libraries and paths for linking npymath lib.
#
#  PYBIND11_FOUND               - was Pybind11 found
#  PYBIND11_VERSION             - the version of Pybind11 found as a string
#  PYBIND11_VERSION_MAJOR       - the major version number of Pybind11
#  PYBIND11_VERSION_MINOR       - the minor version number of Pybind11
#  PYBIND11_VERSION_PATCH       - the patch version number of Pybind11
#  PYBIND11_VERSION_DECIMAL     - e.g. version 1.6.1 is 10601
#  PYBIND11_INCLUDE_DIRS        - path to the Pybind11 include files

#============================================================================
# Copyright 2012 Continuum Analytics, Inc.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#============================================================================

# Finding Pybind11 involves calling the Python interpreter
find_package(pybind11 CONFIG QUIET)
if (pybind11_FOUND)
  set(PYBIND11_FOUND TRUE)
else()
  if(Pybind11_FIND_REQUIRED)
    find_package(PythonInterp REQUIRED)
  else()
    find_package(PythonInterp)
  endif()

  if(NOT PYTHONINTERP_FOUND)
    set(PYBIND11_FOUND FALSE)
  endif()

  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import pybind11 as pb; print(pb.__version__); print(pb.get_include());"
    RESULT_VARIABLE _PYBIND11_SEARCH_SUCCESS
    OUTPUT_VARIABLE _PYBIND11_VALUES
    ERROR_VARIABLE _PYBIND11_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT _PYBIND11_SEARCH_SUCCESS MATCHES 0)
    if(Pybind11_FIND_REQUIRED)
      message(FATAL_ERROR
        "pybind11 import failure:\n${_PYBIND11_ERROR_VALUE}")
    endif()
    set(PYBIND11_FOUND FALSE)
  else()
    set(PYBIND11_FOUND TRUE)
  endif()

  if (PYBIND11_FOUND)
    # Convert the process output into a list
    string(REGEX REPLACE ";" "\\\\;" _PYBIND11_VALUES ${_PYBIND11_VALUES})
    string(REGEX REPLACE "\n" ";" _PYBIND11_VALUES ${_PYBIND11_VALUES})
    list(GET _PYBIND11_VALUES 0 PYBIND11_VERSION)
    list(GET _PYBIND11_VALUES 1 PYBIND11_INCLUDE_DIRS)

    # Make sure all directory separators are '/'
    string(REGEX REPLACE "\\\\" "/" PYBIND11_INCLUDE_DIRS ${PYBIND11_INCLUDE_DIRS})

    # Get the major and minor version numbers
    string(REGEX REPLACE "\\." ";" _PYBIND11_VERSION_LIST ${PYBIND11_VERSION})
    list(GET _PYBIND11_VERSION_LIST 0 PYBIND11_VERSION_MAJOR)
    list(GET _PYBIND11_VERSION_LIST 1 PYBIND11_VERSION_MINOR)
    list(GET _PYBIND11_VERSION_LIST 2 PYBIND11_VERSION_PATCH)
    string(REGEX MATCH "[0-9]*" PYBIND11_VERSION_PATCH ${PYBIND11_VERSION_PATCH})
    math(EXPR PYBIND11_VERSION_DECIMAL
      "(${PYBIND11_VERSION_MAJOR} * 10000) + (${PYBIND11_VERSION_MINOR} * 100) + ${PYBIND11_VERSION_PATCH}")

    find_package_message(PYBIND11
      "Found Pybind11: version \"${PYBIND11_VERSION}\" ${PYBIND11_INCLUDE_DIRS}"
      "${PYBIND11_INCLUDE_DIRS}${PYBIND11_VERSION}")
  endif()
endif()

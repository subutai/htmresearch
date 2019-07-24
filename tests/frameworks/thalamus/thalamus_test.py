# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Tests thalamus module."""

from __future__ import print_function

import unittest

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import skimage
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

from htmresearch.frameworks.thalamus.thalamus import Thalamus
from htmresearch.frameworks.thalamus.thalamus_utils import (
  createLocationEncoder, encodeLocation, trainThalamusLocationsSimple, inferThalamus,
  getUnionLocations, defaultDtype)


# def _randomSDR(numOfBits, size):
#   """
#   Creates a random SDR for the given cell count and SDR size
#   :param numOfBits: Total number of bits
#   :param size: Number of active bits desired in the SDR
#   :return: list with active bits indexes in SDR
#   """
#   return random.sample(xrange(numOfBits), size)


class ThalamusTest(unittest.TestCase):
  """
  Tests for the Thalamus class and associated utilitie.
  """


  def testLearnL6(self):
    """Simple test of the basic interface for L4L2Experiment."""
    t = Thalamus()

    # Learn to associate two L6 SDRs with 2 TRN cells each
    indices1 = t.learnL6Pattern([0, 1, 2, 3, 4, 5], [(0, 0), (2, 3)])
    self.assertEqual(set(indices1), {0, 98})

    indices2 = t.learnL6Pattern([6, 7, 8, 9, 10], [(1, 1), (3, 4)])
    self.assertEqual(set(indices2), {33, 131})

    ff = np.zeros((32, 32))
    ff.reshape(-1)[[8, 9, 98, 99]] = 1.0

    # Should contain no bursting
    ffOutput = inferThalamus(t, [0, 1, 2, 3, 4, 5], ff)
    self.assertEqual(ff.sum(), 4.0)
    self.assertEqual((ff == 2).sum(), 0.0)

    # With lower TRN threshold, should contain two bursting cells
    t.trnActivationThreshold = 2
    ffOutput = inferThalamus(t, [0, 1, 2, 3, 4, 5], ff)
    # self.assertEqual((ff == 2).sum(), 2.0)


if __name__ == "__main__":
  unittest.main()

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

import numpy as np

from htmresearch.frameworks.thalamus.thalamus import Thalamus
from htmresearch.frameworks.thalamus.thalamus_utils import (
  createLocationEncoder, encodeLocation, trainThalamusLocationsSimple, inferThalamus,
  getUnionLocations, defaultDtype, trainThalamus)


class ThalamusTest(unittest.TestCase):
  """
  Tests for the Thalamus class and associated utilitie.
  """


  def testLearnL6Pattern(self):
    """Simple test of the basic learn l6 pattern interface."""
    t = Thalamus()

    # Learn to associate two L6 SDRs with 2 TRN cells each
    indices1 = t.learnL6Pattern([0, 1, 2, 3, 4, 5], [(0, 0), (2, 3)])
    self.assertEqual(set(indices1), {0, 98})
    self.assertEqual(2, t.trnConnections.nSegments())
    self.assertEqual([1, 1], list(t.trnConnections.getSegmentCounts([0, 98])))

    # Learn to associate another two L6 SDRs with 2 TRN cells each
    indices2 = t.learnL6Pattern([6, 7, 8, 9, 10], [(1, 1), (2, 3)])
    self.assertEqual(set(indices2), {33, 98})
    self.assertEqual(4, t.trnConnections.nSegments())
    self.assertEqual([1, 1, 2, 0],
                     list(t.trnConnections.getSegmentCounts([0, 33, 98, 131])))

    # ff = np.zeros((32, 32))
    # ff.reshape(-1)[[8, 9, 98, 99]] = 1.0
    #
    # # Should contain no bursting
    # ffOutput = inferThalamus(t, [0, 1, 2, 3, 4, 5], ff)
    # self.assertEqual(ff.sum(), 4.0)
    # self.assertEqual((ff == 2).sum(), 0.0)
    #
    # # With lower TRN threshold, should contain two bursting cells
    # t.trnActivationThreshold = 2
    # ffOutput = inferThalamus(t, [0, 1, 2, 3, 4, 5], ff)
    # # self.assertEqual((ff == 2).sum(), 2.0)


  def testLearnTRNPatternOnRelayCells(self):
    """Simple test of the basic learn TRN patterns."""
    t = Thalamus()

    # Learn to associate an L6 SDRs and a FF coordinate with 2 relay cells
    # Check that TRN and FF segments on those cells have the same counts
    indices1 = t.learnTRNPatternOnRelayCells([0, 1, 2, 3, 4, 5], (2, 3),
                                             [(0, 0), (2, 3)])
    self.assertEqual(set(indices1), {0, 98})
    self.assertEqual(2, t.relayTRNSegments.nSegments())
    self.assertEqual(2, t.relayFFSegments.nSegments())
    self.assertEqual([1, 1], list(t.relayTRNSegments.getSegmentCounts([0, 98])))
    self.assertEqual([1, 1], list(t.relayFFSegments.getSegmentCounts([0, 98])))

    # Ensure FF segments have the correct input cell
    self.assertEqual(t.ffCellIndex((2, 3)),
                     t.relayFFSegments.matrix.rowNonZeros(0)[0][0])
    self.assertEqual(t.ffCellIndex((2, 3)),
                     t.relayFFSegments.matrix.rowNonZeros(1)[0][0])

    # Learn to associate another L6 SDRs and a FF coordinate with 2 relay cells
    indices2 = t.learnTRNPatternOnRelayCells([6, 7, 8, 9, 10], (1, 1),
                                             [(1, 1), (2, 3)])
    self.assertEqual(set(indices2), {33, 98})
    self.assertEqual(4, t.relayTRNSegments.nSegments())
    self.assertEqual(4, t.relayFFSegments.nSegments())
    self.assertEqual([1, 1, 2, 0],
                     list(t.relayTRNSegments.getSegmentCounts([0, 33, 98, 131])))
    self.assertEqual([1, 1, 2, 0],
                     list(t.relayFFSegments.getSegmentCounts([0, 33, 98, 131])))
    self.assertEqual(t.ffCellIndex((1, 1)),
                     t.relayFFSegments.matrix.rowNonZeros(2)[0][0])
    self.assertEqual(t.ffCellIndex((1, 1)),
                     t.relayFFSegments.matrix.rowNonZeros(3)[0][0])


  def testDeinactivateCells(self):
    """Train the thalamus on two TRN SDRs and then test deinactivation."""
    t = Thalamus(
      trnCellShape=(16, 16),
      relayCellShape=(16, 16),
      inputShape=(16, 16),
      trnThreshold=5,
      relayThreshold=5,
    )

    # Learn to associate the L6 SDR [0, 1, 2, 3, 4] with TRN cells in location
    # (1, 1) - (3, 3) inclusive to get TRN SDR1
    l6SDR1 = [0, 1, 2, 3, 4]
    trnSDR1 = t.learnL6Pattern(l6SDR1,
                                [
                                  (1, 1), (1, 2), (1, 3),
                                  (2, 1), (2, 2), (2, 3),
                                  (3, 1), (3, 2), (3, 3),
                                ])
    # Train the relay cell at (2, 2) to associate TRN SDR1 with FF location (2, 2)
    relayIndices1 = t.learnTRNPatternOnRelayCells(trnSDR1, (2, 2), [(2, 2)])

    # Learn to associate the L6 SDR [5, 6, 7, 8, 9] with TRN cells in location
    # (11, 11) - (13, 13) inclusive to get TRN SDR2
    l6SDR2 = [5, 6, 7, 8, 9]
    trnSDR2 = t.learnL6Pattern(l6SDR2,
                                [
                                  (11, 11), (11, 12), (11, 13),
                                  (12, 11), (12, 12), (12, 13),
                                  (13, 11), (13, 12), (13, 13),
                                ])

    # Train the relay cell at (12, 12) to associate TRN SDR2 with FF location (12, 12)
    relayIndices2 = t.learnTRNPatternOnRelayCells(trnSDR2, (12, 12), [(12, 12)])

    # Deinactivate using each L6 SDR and ensure that the correct relay cells become
    # burst ready
    t.deInactivateCells(l6SDR1)
    self.assertEqual(set(t.burstReadyCellIndices), set(relayIndices1))
    self.assertEqual(set(t.burstReadyCells.reshape(-1).nonzero()[0]),
                     set(relayIndices1))

    t.reset()
    t.deInactivateCells(l6SDR2)
    self.assertEqual(set(t.burstReadyCellIndices), set(relayIndices2))
    self.assertEqual(set(t.burstReadyCells.reshape(-1).nonzero()[0]),
                     set(relayIndices2))


  def testTrainThalamus(self):
    """Train thalamus utility."""
    t = Thalamus(
      trnCellShape=(16, 16),
      relayCellShape=(16, 16),
      inputShape=(16, 16),
      trnThreshold=5,
      relayThreshold=5,
    )
    encoder = createLocationEncoder(t, w=11)
    trainThalamus(t, encoder, windowSize=3)

    output = np.zeros(encoder.getWidth(), dtype=defaultDtype)
    l6Input = encodeLocation(encoder, 8, 8, output)

    ffInput = np.zeros((16, 16))
    ffInput[:] = 0
    ffInput[10:20, 10:20] = 1

    t.reset()
    t.deInactivateCells(l6Input)
    ffOutput = t.computeFeedForwardActivity(ffInput)


if __name__ == "__main__":
  unittest.main()

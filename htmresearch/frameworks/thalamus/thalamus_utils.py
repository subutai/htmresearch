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

"""

Utility functions

"""

from __future__ import print_function

import numpy as np

from nupic.encoders.base import defaultDtype
from nupic.encoders.coordinate import CoordinateEncoder


def createLocationEncoder(t, w=15):
  """
  A default coordinate encoder for encoding locations into sparse
  distributed representations.
  """
  encoder = CoordinateEncoder(name="positionEncoder", n=t.l6CellCount, w=w)
  return encoder


def encodeLocation(encoder, x, y, output, radius=5):
  # Radius of 7 or 8 gives an overlap of about 8 or 9 with neighoring pixels
  # Radius of 5 about 3
  encoder.encodeIntoArray((np.array([x * radius, y * radius]), radius), output)
  return output.nonzero()[0]


def trainThalamusLocationsSimple(t, encoder):
  print("Training TRN cells on location SDRs")
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  # Train the TRN cells to respond to SDRs representing locations
  for y in range(0, t.trnHeight):
    for x in range(0, t.trnWidth):
      t.learnL6Pattern(encodeLocation(encoder, x, y, output),
                       [(x, y)])


def inferThalamus(t, l6Input, ffInput):
  """
  Compute the effect of this feed forward input given the specific L6 input.

  :param t: instance of Thalamus
  :param l6Input:
  :param ffInput: a numpy array of 0's and 1's
  :return:
  """
  print("\n-----------")
  t.reset()
  t.deInactivateCells(l6Input)
  ffOutput = t.computeFeedForwardActivity(ffInput)
  # print("L6 input:", l6Input)
  # print("Active TRN cells: ", t.activeTRNCellIndices)
  # print("Burst ready relay cells: ", t.burstReadyCellIndices)
  return ffOutput


def getUnionLocations(encoder, x, y, r, step=1):
  """
  Return a union of location encodings that correspond to the union of all locations
  within the specified circle.
  """
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)
  locations = set()
  for dx in range(-r, r+1, step):
    for dy in range(-r, r+1, step):
      if dx*dx + dy*dy <= r*r:
        e = encodeLocation(encoder, x+dx, y+dy, output)
        locations = locations.union(set(e))

  return locations


def trainThalamus(t, encoder, windowSize=5):
  """
  Train the thalamus to recognize location SDRs.

  For each location (wx, wy), we create an L6 SDR that represents that location.

  We then train a set of TRN cells located in a window around (wx, wy) to recognize
  that SDR.  Each TRN cell will contain a dendrite specific to (wx, wy). So we get a TRN
  SDR that represents the location (wx, wy).

  We also train a set of relay cells located in a window around (wx, wy) to recognize
  that TRN SDR.  Each relay cell will contain a dendrite that recognizes the TRN SDR
  corresponding to (wx, wy).

  At the end, each (wx, wy), will activate a set of TRN cells. This set of TRN cells
  will activate a set of relay cells.

  :param t:
  :param encoder:
  :param windowSize:
  :return:
  """
  print("Training TRN cells on location SDRs")
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  # Train the TRN cells to respond to SDRs representing locations
  for wy in range(0, t.trnHeight):
    print(wy)
    for wx in range(0, t.trnWidth):
      l6LocationSDR = encodeLocation(encoder, wx, wy, output)
      
      # Train TRN cells located around wx,wy to recognize this SDR. The set
      # of TRN cells will represent a TRN SDR for this locationn.
      # TODO: convert loop to list comprehension
      trnSDRIndices = set()
      trnCellsToLearnOn = []
      for x in range(wx-windowSize, wx+windowSize):
        for y in range(wy - windowSize, wy + windowSize):
          if x >= 0 and x < t.trnWidth and y >= 0 and y < t.trnHeight:
            trnCellsToLearnOn.append((x, y))

      indices = t.learnL6Pattern(l6LocationSDR, trnCellsToLearnOn)
      trnSDRIndices = list(trnSDRIndices.union(set(indices)))

      # Train relay cells located around wx, wy to recognize the TRN SDR.
      relayCellsToLearnOn = []
      relaySDRIndices = set()
      for x in range(wx-windowSize, wx+windowSize):
        for y in range(wy - windowSize, wy + windowSize):
          if x >= 0 and x < t.trnWidth and y >= 0 and y < t.trnHeight:
            relayCellsToLearnOn.append((x, y))

      indices = t.learnTRNPatternOnRelayCells(
                                  trnSDRIndices, (wx, wy), relayCellsToLearnOn)
      relaySDRIndices = relaySDRIndices.union(set(indices))


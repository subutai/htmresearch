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

An implementation of thalamic control and routing as proposed in the Cosyne
submission:

  A dendritic mechanism for dynamic routing and control in the thalamus
           Carmen Varela & Subutai Ahmad

"""

from __future__ import print_function

import numpy as np

from nupic.bindings.math import Random, SparseMatrixConnections



class Thalamus(object):
  """
  A simple discrete time thalamus.  This thalamus has a 2D TRN layer and a 2D
  relay cell layer. L6 cells project to the dendrites of TRN cells - these
  connections are learned. TRN cells project to the dendrites of relay cells in
  a fixed fan-out pattern. A 2D feed forward input source projects to the relay
  cells in a fixed fan-out pattern.

  The output of the thalamus is the activity of each relay cell. This activity
  can be in one of three states: inactive, active (tonic), and active (burst).

  TRN cells control whether the relay cells will burst. If any dendrite on a TRN
  cell recognizes the current L6 pattern, it de-inactivates the T-type CA2+
  channels on the dendrites of any relay cell it projects to. These relay cells
  are then in "burst-ready mode".

  Feed forward activity is in the form of a binary vector corresponding to
  active/spiking axons (e.g. from ganglion cells). Any relay cells that receive
  input from an axon will either output tonic or burst activity depending on the
  state of the T-type CA2+ channels on their dendrites. Relay cells that don't
  receive input will remain inactive, regardless of their dendritic state.

  This class assumes the shape of the feed forward input layer and the relay layer
  are identical and aligned. Each relay cell receives input from a 3x3 block of
  feed forward input neurons.

  Usage:

    1. Train the TRN and relay cells on a bunch of L6 patterns using
       learnL6Pattern() and learnTRNPatternOnRelayCells()

    2. De-inactivate relay cells by sending in an L6 pattern: deInactivateCells()

    3. Compute feed forward activity for an input: computeFeedForwardActivity()

    4. reset()

    5. Goto 2


  Currently we have L6 patterns being recognized by TRN cells, which deinactivate relay
  cell dendrites. The idea is that the L6 to relay cell connections act as a reset
  signal and bring relay cells back to inactivated tonic mode. The reset is likely to be
  a general broad signal. If there is a global reset, then all relay cell dendrites
  would have the same set of connections.

  This assumes that areas of interest are put into burst-ready mode. Alternatively we've
  heard the opposite theory, that areas of non-interest are in burst ready mode. This
  allows the neocortex to process attended areas, but something unusual happening
  outside this area will cause attention to shift. In this case, tonic mode is the mode
  used in attended areas, and the L6 to relay cell dendrites would be recognizing the
  pattern. TRN cells would act more broadly to put relay cells into burst mode
  everywhere except the attended areas.


  There are still a few TODOs:

  TODO: reset mechanism needs to be implemented.

  TODO: incorporate actual convergence numbers from Convergence CT-TRN-TC.
  Kultas-Ilinsky & Ilinsky 1991; Harting et al 1991, Norita & Katoh 1987, Cucchiaro 1991
  (refs from Sato 1997)

  TODO: currently the # TRN cells = # relay cells = # input axons. Remove this
  restriction. (Low priority)

  TODO: implement an offset relationship. Bursts indicate a mismatch between prediction
  and stimulus. Can this explain the low frequency of bursting?

  """

  def __init__(self,
               trnCellShape=(32, 32),
               relayCellShape=(32, 32),
               inputShape=(32, 32),
               l6CellCount=1024,
               trnThreshold=10,
               relayThreshold=10,
               seed=42):
    """

    :param trnCellShape:
      a 2D shape for the TRN

    :param relayCellShape:
      a 2D shape for the relay cells

    :param l6CellCount:
      number of L6 cells

    :param trnThreshold:
      dendritic threshold for TRN cells. This is the min number of active L6
      cells on a dendrite for the TRN cell to recognize a pattern on that
      dendrite.

    :param relayThreshold:
      dendritic threshold for relay cells. This is the min number of active TRN
      cells on a dendrite for the relay cell to recognize a pattern on that
      dendrite.

    :param seed:
        Seed for the random number generator.
    """

    # Shapes of TRN cell layer, relay cell layer, and feed forward input layer
    self.trnCellShape = trnCellShape
    self.trnWidth = trnCellShape[0]
    self.trnHeight = trnCellShape[1]

    self.relayCellShape = relayCellShape
    self.relayWidth = relayCellShape[0]
    self.relayHeight = relayCellShape[1]

    self.inputShape = inputShape
    self.inputWidth = inputShape[0]
    self.inputHeight = inputShape[1]


    self.l6CellCount = l6CellCount
    self.seed = seed
    self.rng = Random(seed)
    self.trnActivationThreshold = trnThreshold

    self.trnConnections = SparseMatrixConnections(
      trnCellShape[0]*trnCellShape[1], l6CellCount)

    self.relayTRNSegmentThreshold = relayThreshold
    self.relayTRNSegments = SparseMatrixConnections(
      relayCellShape[0]*relayCellShape[1],
      trnCellShape[0]*trnCellShape[1])

    self.relayFFSegmentThreshold = 1
    self.relayFFSegments = SparseMatrixConnections(
      relayCellShape[0]*relayCellShape[1],
      inputShape[0]*inputShape[1])

    # Initialize/reset variables that are updated with calls to compute
    self.reset()


  def learnL6Pattern(self, l6Pattern, cellsToLearnOn):
    """
    Learn the given l6Pattern on TRN cell dendrites. The TRN cells to learn
    are given in cellsTeLearnOn. Each of these cells will learn this pattern on
    a single dendritic segment.

    :param l6Pattern:
      An SDR from L6. List of indices corresponding to L6 cells.

    :param cellsToLearnOn:
      Each cell index is (x,y) corresponding to the TRN cells that should learn
      this pattern. For each cell, create a new dendrite that stores this
      pattern. The SDR is stored on this dendrite

    :return: the list of TRN cell indices that learned this pattern
    """
    cellIndices = [self.trnCellIndex(x) for x in cellsToLearnOn]
    newSegments = self.trnConnections.createSegments(cellIndices)
    self.trnConnections.growSynapses(newSegments, l6Pattern, 1.0)

    # At this point we would want to train relay cells around each TRN cell

    # print("Learning L6 SDR:", l6Pattern,
    #       "new segments: ", newSegments,
    #       "cells:", self.trnConnections.mapSegmentsToCells(newSegments))

    return cellIndices


  def learnTRNPatternOnRelayCells(self, trnSDRIndices, ffCoord, cellsToLearnOn):
    """
    Learn the given TRN pattern on relay cell dendrites. The dendrite also associates
    with the given feed forward coordinate. The relay cells to learn are given in
    cellsToLearnOn.

    Implementation note:

    Each of these cells will learn the TRN pattern on a single segment in
    relayTRNSegments. Each of these cells will also learn the associated feed forward
    input (single feed forward axon) onto a segment in relayFFSegments. Each dendrite on
    a relay cell is represented by one segment in relayTRNSegments plus one segment in
    relayFFSegments. We thus enforce a one to one correspondence between these two lists
    of segments.

    This separation into two lists is required to independently identify whether a TRN
    pattern is detected, whether a FF input is detected, and then whether the cell is
    responding in bursting or tonic mode. Case 1) If a TRN pattern is detected on a
    segment, that dendrite is in de-inactivated mode. If the associated dendrite
    contains a segment that also recognizes the FF input, the dendrite will burst and
    cause the cell to respond with burst firing. Case 2) Otherwise, if a dendrite
    detects a FF input, but its corresponding dendrite is not de-inactivated (and no
    other dendrite is bursting) the cell will respond with tonic firing. Case 3)
    Otherwise (no FF input detected on any dendrite) the cell will be silent.

    :param trnSDRIndices:
      An SDR from TRN. List of indices corresponding to TRN cells, [idx1, idx2, ...]

    :param ffCoord:
      The coordinate of the associated feed forward axon, (x,y)

    :param cellsToLearnOn:
      List of cell coordinates [(x, y), ...]. Each cell coord corresponds to a relay
      cell that should learn this pattern. For each cell, create a new dendrite that
      learns this trnSDR and the ffCoord.

    :return: the list of relay cell indices that learned this pattern
    """
    assert len(ffCoord)==2
    cellIndices = [self.relayCellIndex(x) for x in cellsToLearnOn]

    # We create exactly the same number of segments to hold TRN connections and FF
    # connections respectively. We enforce the same segment indices for TRN and FF
    # connections.
    newTRNSegments = self.relayTRNSegments.createSegments(cellIndices)
    newFFSegments = self.relayFFSegments.createSegments(cellIndices)
    assert (np.array_equal(newFFSegments, newTRNSegments))
    self.relayTRNSegments.growSynapses(newTRNSegments, trnSDRIndices, 1.0)

    # Now we want the FF dendrites to create a connection to this FF pattern.
    ffIndex = [self.ffCellIndex(ffCoord)]
    self.relayFFSegments.growSynapses(newFFSegments, ffIndex, 1.0)

    return cellIndices


  def deInactivateCells(self, l6Input):
    """
    Activate trnCells according to the l6Input. These in turn will impact 
    bursting mode in relay cells that are connected to these trnCells.
    Given the feedForwardInput, compute which cells will be silent, tonic,
    or bursting.
    
    :param l6Input:

    :return: nothing
    """

    # Figure out which TRN cells recognize the L6 pattern.
    self.trnOverlaps = self.trnConnections.computeActivity(l6Input, 0.5)
    self.activeTRNSegments = np.flatnonzero(
      self.trnOverlaps >= self.trnActivationThreshold)
    self.activeTRNCellIndices = self.trnConnections.mapSegmentsToCells(
      self.activeTRNSegments)

    # for s, idx in zip(self.activeTRNSegments, self.activeTRNCellIndices):
    #   print(self.trnOverlaps[s], idx, self.trnIndextoCoord(idx))

    # Figure out which relay cells have dendrites in de-inactivated state
    self.relayTRNSegmentOverlaps = self.relayTRNSegments.computeActivity(
      self.activeTRNCellIndices, 0.5
    )
    self.activeRelaySegments = np.flatnonzero(
      self.relayTRNSegmentOverlaps >= self.relayTRNSegmentThreshold)
    self.burstReadyCellIndices = self.relayTRNSegments.mapSegmentsToCells(
      self.activeRelaySegments)

    # relayTRNSegmentOverlaps is a numpy array containing the overlap score
    # with the TRN input for each TRN segment.
    #
    # activeRelaySegments is a numpy array holding segment indices for those
    # segments that are de-inactivated
    #
    # burstReadyCellIndices contains the cell index for each segment in activeRelaySegments

    self.burstReadyCells.reshape(-1)[self.burstReadyCellIndices] = 1


  def computeFeedForwardActivity(self, feedForwardInput, tonicLevel=0.4):
    """
    Activate trnCells according to the l6Input. These in turn will impact
    bursting mode in relay cells that are connected to these trnCells.
    Given the feedForwardInput, compute which cells will be silent, tonic,
    or bursting.

    :param feedForwardInput:
      a numpy matrix of shape relayCellShape containing 0's and 1's

    :return:
      Relay cell activity as a numpy matrix.
      feedForwardInput is modified to contain 0, 1, or 2. A "2" indicates
      bursting cells.
    """
    ff = feedForwardInput.copy()
    ffLocations = ff.nonzero()
    ffindices = [self.ffCellIndex(c) for c in ffLocations]
    self.ffOverlaps = self.relayFFSegments.computeActivity(ffindices, 0.5)
    self.activeFFSegments = np.flatnonzero(self.ffOverlaps >= 1)

    # Now, any cell with an activeFFSegment will respond in tonic mode

    # Now, those cells where activeFFSegments and activeRelaySegments match
    # up will respond in burst mode, and override tonic mode.

    # # For each relay cell, see if any of its FF inputs are active.
    # for x in range(self.relayWidth):
    #   for y in range(self.relayHeight):
    #     inputCells = self._preSynapticFFCells(x, y)
    #     for idx in inputCells:
    #       if feedForwardInput[idx] != 0:
    #         ff[x, y] = 1.0
    #         continue
    #
    # # If yes, and it is in burst mode, this cell bursts
    # # If yes, and it is not in burst mode, then we just get tonic input.
    #
    # # ff += self.burstReadyCells * ff
    ff2 = ff * tonicLevel + self.burstReadyCells * ff
    return ff2


  # OLD WAY
  # def computeFeedForwardActivity(self, feedForwardInput, tonicLevel=0.4):
  #   """
  #   Activate trnCells according to the l6Input. These in turn will impact
  #   bursting mode in relay cells that are connected to these trnCells.
  #   Given the feedForwardInput, compute which cells will be silent, tonic,
  #   or bursting.
  #
  #   :param feedForwardInput:
  #     a numpy matrix of shape relayCellShape containing 0's and 1's
  #
  #   :return:
  #     Relay cell activity as a numpy matrix.
  #     feedForwardInput is modified to contain 0, 1, or 2. A "2" indicates
  #     bursting cells.
  #   """
  #   ff = feedForwardInput.copy()
  #   # For each relay cell, see if any of its FF inputs are active.
  #   for x in range(self.relayWidth):
  #     for y in range(self.relayHeight):
  #       inputCells = self._preSynapticFFCells(x, y)
  #       for idx in inputCells:
  #         if feedForwardInput[idx] != 0:
  #           ff[x, y] = 1.0
  #           continue
  #
  #   # If yes, and it is in burst mode, this cell bursts
  #   # If yes, and it is not in burst mode, then we just get tonic input.
  #
  #   # ff += self.burstReadyCells * ff
  #   ff2 = ff * tonicLevel + self.burstReadyCells * ff
  #   return ff2


  def reset(self):
    """
    Set everything back to zero
    """
    self.trnOverlaps = []
    self.activeTRNSegments = []
    self.activeTRNCellIndices = []
    self.relayTRNSegmentOverlaps = []
    self.activeRelaySegments = []
    self.burstReadyCellIndices = []
    self.burstReadyCells = np.zeros((self.relayWidth, self.relayHeight))


  def trnCellIndex(self, coord):
    """
    Map a 2D coordinate to 1D cell index.

    :param coord: a 2D coordinate

    :return: integer index
    """
    return coord[1] * self.trnWidth + coord[0]


  def trnIndextoCoord(self, i):
    """
    Map 1D cell index to a 2D coordinate

    :param i: integer 1D cell index

    :return: (x, y), a 2D coordinate
    """
    x = i % self.trnWidth
    y = i / self.trnWidth
    return x, y


  def relayCellIndex(self, coord):
    """
    Map a 2D coordinate to 1D cell index.

    :param coord: a 2D coordinate

    :return: integer index
    """
    return coord[1] * self.relayWidth + coord[0]


  def relayIndextoCoord(self, i):
    """
    Map 1D cell index to a 2D coordinate

    :param i: integer 1D cell index

    :return: (x, y), a 2D coordinate
    """
    x = i % self.relayWidth
    y = i / self.relayWidth
    return x, y


  def ffCellIndex(self, coord):
    """
    Map a 2D coordinate to 1D cell index.

    :param coord: a 2D coordinate

    :return: integer index
    """
    return coord[1] * self.inputWidth + coord[0]


  def ffIndextoCoord(self, i):
    """
    Map 1D cell index to a 2D coordinate

    :param i: integer 1D cell index

    :return: (x, y), a 2D coordinate
    """
    x = i % self.inputWidth
    y = i / self.inputWidth
    return x, y


  def _initializeTRNToRelayCellConnections(self):
    """
    Initialize TRN to relay cell connectivity. For each relay cell, create a
    dendritic segment for each TRN cell it connects to.
    """
    for x in range(self.relayWidth):
      for y in range(self.relayHeight):

        # Create one dendrite for each trn cell that projects to this relay cell
        # This dendrite contains one synapse corresponding to this TRN->relay
        # connection.
        relayCellIndex = self.relayCellIndex((x,y))
        trnCells = self._preSynapticTRNCells(x, y)
        for trnCell in trnCells:
          newSegment = self.relayTRNSegments.createSegments([relayCellIndex])
          self.relayTRNSegments.growSynapses(newSegment,
                                             [self.trnCellIndex(trnCell)], 1.0)


  # OBSOLETE
  # def _learnTRNToRelayCellConnections(self, relayCellsToLearnOn, trnIndices):
  #   """
  #   For each relay cell, create a dendritic segment with connections to each of
  #   the given TRN cells.
  #
  #   :param relayCellsToLearnOn: list of relay cells that will learn,
  #     specified as (x,y) coordinates.
  #
  #   :param trnIndices: cell indices of the TRN cells to connect to.
  #
  #   """
  #   for x in range(self.relayWidth):
  #     for y in range(self.relayHeight):
  #
  #       # Create one dendrite for each trn cell that projects to this relay cell
  #       # This dendrite contains one synapse corresponding to this TRN->relay
  #       # connection.
  #       relayCellIndex = self.relayCellIndex((x,y))
  #       trnCells = self._preSynapticTRNCells(x, y)
  #       for trnCell in trnCells:
  #         newSegment = self.relayTRNSegments.createSegments([relayCellIndex])
  #         self.relayTRNSegments.growSynapses(newSegment,
  #                                            [self.trnCellIndex(trnCell)], 1.0)


  def _initializeRelayCellDendrites(self):
    """
    Initialize relay cell dendrites. If we assume that tau TRN cells connect to
    a given relay cell, and gamma feed-forward (FF) axons (e.g. ganglion cell
    axons) connect to each relay cell, we create tau * gamma dendritic segments
    on the relay cell. Each dendrite will have one of the TRN connections, and
    one of the FF connections.

    """
    pass


  def _preSynapticTRNCells(self, i, j):
    """
    Given a relay cell at the given coordinate, return a list of the (x,y)
    coordinates of all TRN cells that project to it. This assumes a 3X3 fan-in.

    :param i, j: relay cell Coordinates

    :return:
    """
    xmin = max(i - 1, 0)
    xmax = min(i + 2, self.trnWidth)
    ymin = max(j - 1, 0)
    ymax = min(j + 2, self.trnHeight)
    trnCells = [
      (x, y) for x in range(xmin, xmax) for y in range(ymin, ymax)
    ]

    return trnCells


  def _preSynapticFFCells(self, i, j):
    """
    Given a relay cell at the given coordinate, return a list of the (x,y) coordinates
    of all feed forward input cells that project to it. This assumes a 3X3 fan-in.

    :param i, j: relay cell Coordinates

    :return:
    """
    xmin = max(i - 1, 0)
    xmax = min(i + 2, self.inputWidth)
    ymin = max(j - 1, 0)
    ymax = min(j + 2, self.inputHeight)
    inputCells = [
      (x, y) for x in range(xmin, xmax) for y in range(ymin, ymax)
    ]

    return inputCells


/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree.impl

import org.apache.spark.ml.tree._
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.ImpurityStats

/** Object exposing methods for local training of decision trees */
private[ml] object LocalDecisionTree {

  /**
   * Fully splits the passed-in node on the provided local dataset.
   * TODO(smurching): Accept a seed for feature subsampling
   *
   * @param node LearningNode to split
   */
  def fitNode(
      input: Array[BaggedPoint[TreePoint]],
      node: LearningNode,
      metadata: DecisionTreeMetadata,
      splits: Array[Array[Split]]): Node = {

    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    // TODO: Check that learning works when maxDepth > 0 but learning stops at 1 node (because of
    //       other parameters).
    if (metadata.maxDepth == 0) {
      val impurityAggregator: ImpurityAggregatorSingle =
        input.aggregate(metadata.createImpurityAggregator())(
          (agg, lp) => agg.update(lp.datum.label, 1.0),
          (agg1, agg2) => agg1.add(agg2))
      val impurityCalculator = impurityAggregator.getCalculator
      return new LeafNode(LocalDecisionTreeUtils.getPredict(impurityCalculator).predict,
        impurityCalculator.calculate(), impurityCalculator)
    }

    // Prepare column store.
    //   Note: rowToColumnStoreDense checks to make sure numRows < Int.MaxValue.
    val colStoreInit: Array[Array[Int]]
      = LocalDecisionTreeUtils.rowToColumnStoreDense(input.map(_.datum.binnedFeatures))
    val labels = input.map(_.datum.label)

    // Train classifier if numClasses is between 1 and 32, otherwise fit a regression model
    // on the dataset
    if (metadata.numClasses > 1 && metadata.numClasses <= 32) {
      throw new UnsupportedOperationException("Local training of a decision tree classifier is" +
        "unsupported; currently, only regression is supported")
    } else {
      // TODO(smurching): Pass an array of instanceWeights extracted from the input BaggedPoint?
      // Also, pass seed for feature subsampling
      trainRegressor(node, colStoreInit, labels, metadata, splits)
    }
  }

  /**
   * Locally fits a decision tree regressor.
   * TODO(smurching): Logic for fitting a classifier & regressor seems the same; only difference
   * is impurity metric. Use the same logic for fitting a classifier?
   *
   * @param rootNode Node to fit on the passed-in dataset
   * @param colStoreInit Array of columns of training data
   * @param metadata Metadata object
   * @param splits splits(i) = Array of possible splits for feature i
   * @return
   */
  def trainRegressor(
      rootNode: LearningNode,
      colStoreInit: Array[Array[Int]],
      labels: Array[Double],
      metadata: DecisionTreeMetadata,
      splits: Array[Array[Split]]): Node = {

    // Sort each column by feature values.
    val colStore: Array[FeatureVector] = colStoreInit.zipWithIndex.map { case (col, featureIndex) =>
      val featureArity: Int = metadata.featureArity.getOrElse(featureIndex, 0)
      FeatureVector.fromOriginal(featureIndex, featureArity, col)
    }

    val numRows = colStore.headOption match {
      case None => 0
      case Some(column) => column.values.length
    }

    // Create a new PartitionInfo describing the status of our partially-trained subtree
    // at each iteration of training
    var partitionInfo: PartitionInfo = new PartitionInfo(colStore,
      nodeOffsets = Array[(Int, Int)]((0, numRows)), activeNodes = Array(rootNode))

    // Iteratively learn, one level of the tree at a time.
    // Note: We do not use node IDs.
    var currentLevel = 0
    var doneLearning = false

    while (currentLevel < metadata.maxDepth && !doneLearning) {
      // Splits each active node if possible, returning an array of new active nodes
      val activeNodes: Array[LearningNode] =
        computeBestSplits(partitionInfo, labels, metadata, splits)
      // Filter active node periphery by impurity.
      val estimatedRemainingActive = activeNodes.count(_.stats.impurity > 0.0)
      // TODO: Check to make sure we split something, and stop otherwise.
      doneLearning = currentLevel + 1 >= metadata.maxDepth || estimatedRemainingActive == 0
      if (!doneLearning) {
        // Compute bit vector (1 bit/instance) indicating whether each instance goes left/right
        // Only requires partitionInfo since all active nodes have already been split
        val bitset = LocalDecisionTreeUtils.computeBitVector(partitionInfo, numRows, splits)
        // Obtain a new partitionInfo instance describing our current training status
        partitionInfo = partitionInfo.update(bitset, activeNodes, labels, metadata)
        assert(partitionInfo.nodeOffsets.length == partitionInfo.activeNodes.length)
      }
      currentLevel += 1
    }

    // Done with learning
    rootNode.toNode
  }

  /**
   * Find the best splits for all active nodes.
   *  - For each feature, select the best split for each node.
   *
   * @return  Array of new active nodes formed by splitting the current set of active nodes.
   */
  private[impl] def computeBestSplits(
      partitionInfo: PartitionInfo,
      labels: Array[Double],
      metadata: DecisionTreeMetadata,
      splits: Array[Array[Split]]): Array[LearningNode] = {
    // For each node, select the best split across all features
    partitionInfo match {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[(Int, Int)],
        activeNodes: Array[LearningNode]) => {

        // Iterate over the active nodes in the current level.
        activeNodes.zipWithIndex.flatMap { case (node: LearningNode, nodeIndex: Int) =>
          // Features for the current node start at fromOffset and end at toOffset
          val (from, to) = nodeOffsets(nodeIndex)
          // Get the impurity aggregator for the current node
          // TODO(smurching): Test on empty data, see if this still works
          val fullImpurityAgg = LocalDecisionTreeUtils.getImpurity(columns(0), from, to, metadata, labels)
          // TODO(smurching): In PartitionInfo, keep track of which features are associated
          // with which nodes and subsample here (filter columns)
          val splitsAndStats = columns.map { col =>
            chooseSplit(col, labels, from, to, fullImpurityAgg, metadata, splits)
          }
          // For the current node, we pick the feature whose best split results in maximal gain
          val (bestSplit, bestStats) = splitsAndStats.maxBy(_._2.gain)
          // Split current node, get an iterator over its children
          splitIfPossible(node, metadata, bestStats, bestSplit)
        }
      }
    }
  }

  /**
   * Splits the passed-in node if permitted by the parameters of the learning algorithm,
   * returning an iterator over its children. Returns an empty array if node could not be split.
   *
   * @param metadata Metadata containing parameters of the learning algorithm
   * @param stats Label impurity stats associated with the current node
   */
  private[impl] def splitIfPossible(
      node: LearningNode,
      metadata: DecisionTreeMetadata,
      stats: ImpurityStats,
      split: Option[Split]): Iterator[LearningNode] = {

    val leftCount = stats.leftImpurityCalculator.count
    val rightCount = stats.rightImpurityCalculator.count
    val shouldSplit = split.nonEmpty && stats.gain > metadata.minInfoGain &&
      leftCount >= metadata.minInstancesPerNode && rightCount >= metadata.minInstancesPerNode
    if (shouldSplit) {
      // Split node and return an iterator over its children
      doSplit(node, split, stats)
      Iterator(node.leftChild.get, node.rightChild.get)
    } else {
      // If our node has non-zero impurity, set the node's stats to invalid values to indicate that
      // it could not be split due to the parameters of the learning algorithms
      if (stats.impurity != 0) {
        node.stats = ImpurityStats.getInvalidImpurityStats(stats.impurityCalculator)
      }
      node.isLeaf = true
      Iterator()
    }
  }

  /**
   * Splits the passed-in node. This method returns nothing, but modifies the passed-in node
   * by updating its split and stats members.
   *
   * @param split Split to associate with the passed-in node
   * @param stats Label impurity statistics to associate with the passed-in node
   */
  private[impl] def doSplit(
      node: LearningNode,
      split: Option[Split],
      stats: ImpurityStats): Unit = {
    node.leftChild = Some(LearningNode(id = -1, isLeaf = false,
      ImpurityStats.getEmptyImpurityStats(stats.leftImpurityCalculator)))
    node.rightChild = Some(LearningNode(id = -1, isLeaf = false,
      ImpurityStats.getEmptyImpurityStats(stats.rightImpurityCalculator)))
    node.split = split
    node.isLeaf = false
    node.stats = stats
  }

  /**
   * Choose the best split for a feature at a node.
   * TODO: Return null or None when the split is invalid, such as putting all instances on one
   *       child node.
   *
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.
   */
  private[impl] def chooseSplit(
      col: FeatureVector,
      labels: Array[Double],
      fromOffset: Int,
      toOffset: Int,
      fullImpurityAgg: ImpurityAggregatorSingle,
      metadata: DecisionTreeMetadata,
      splits: Array[Array[Split]]): (Option[Split], ImpurityStats) = {
    if (col.isCategorical) {
      if (metadata.isUnordered(col.featureIndex)) {
        val catSplits: Array[CategoricalSplit] = splits(col.featureIndex)
          .map(_.asInstanceOf[CategoricalSplit])
        chooseUnorderedCategoricalSplit(col.featureIndex, col.values, col.indices, labels,
          fromOffset, toOffset, col.featureArity, metadata, catSplits, fullImpurityAgg)
      } else {
        chooseOrderedCategoricalSplit(col.featureIndex, col.values, col.indices,
          labels, fromOffset, toOffset, col.featureArity, metadata, fullImpurityAgg)
      }
    } else {
      chooseContinuousSplit(col.featureIndex, col.values, col.indices, labels, fromOffset, toOffset,
        metadata, splits, fullImpurityAgg)
    }
  }

  /**
   * Given integers (from, to) describing the starting/ending indices of feature values for
   * an active node, returns an array of impurity aggregators containing label statistics
   * for each bin/category for the current (node, feature).
   *
   * @param metadata Metadata object containing parameters of the learning algorithm
   * @param values Array of values for the current feature
   * @param labels Array of labels
   * @param indices indices(i) = row index of value at values(i)
   */
  private[impl] def getAggStats(
      metadata: DecisionTreeMetadata,
      featureIndex: Int,
      from: Int,
      to: Int,
      values: Array[Int],
      labels: Array[Double],
      indices: Array[Int]): Array[ImpurityAggregatorSingle] = {
    // Build an array of ImpurityAggregators for each bin of the current feature
    val aggStats = Array.tabulate[ImpurityAggregatorSingle](metadata.numBins(featureIndex))(
      _ => metadata.createImpurityAggregator())
    from.until(to).foreach { i =>
      val cat = values(i)
      val label = labels(indices(i))
      aggStats(cat).update(label)
    }
    aggStats
  }

  /**
   * TODO(smurching): update doc, also share this between RF code and local code
   * @param metadata
   * @param categoryStats
   * @return
   */
  private[impl] def getCentroid(
      metadata: DecisionTreeMetadata,
      categoryStats: ImpurityAggregatorSingle): Double = {

    if (categoryStats.getCount != 0) {
      if (metadata.isMulticlass) {
        // multiclass classification
        // For categorical variables in multiclass classification,
        // the bins are ordered by the impurity of their corresponding labels.
        categoryStats.getCalculator.calculate()
      } else if (metadata.isClassification) {
        // binary classification
        // For categorical variables in binary classification,
        // the bins are ordered by the count of class 1.
        categoryStats.stats(1)
      } else {
        // regression
        // For categorical variables in regression and binary classification,
        // the bins are ordered by the prediction.
        categoryStats.getCalculator.predict
      }
    } else {
      Double.MaxValue
    }
  }

  /**
   * TODO(smurching): Update doc
   * Calculate the impurity statistics for a given (feature, split) based upon left/right
   * aggregates.
   */
  private def calculateImpurityStats(
      fullImpurityAgg: ImpurityAggregatorSingle,
      leftAgg: ImpurityAggregatorSingle,
      rightAgg: ImpurityAggregatorSingle,
      metadata: DecisionTreeMetadata): ImpurityStats = {

    val fullCalc = fullImpurityAgg.getCalculator
    val leftCalc = leftAgg.getCalculator
    val rightCalc = rightAgg.getCalculator
    calculateImpurityStats(fullCalc, leftCalc, rightCalc, metadata)
  }

  /**
   * TODO(smurching): Update doc
   * Calculate the impurity statistics for a given (feature, split) based upon left/right
   * aggregates.
   *
   * @param stats the recycle impurity statistics for this feature's all splits,
   *              only 'impurity' and 'impurityCalculator' are valid between each iteration
   * @param leftCalc left node aggregates for this (feature, split)
   * @param rightCalc right node aggregate for this (feature, split)
   * @param metadata learning and dataset metadata for DecisionTree
   * @return Impurity statistics for this (feature, split)
   */
  private def calculateImpurityStats(
      fullCalc: ImpurityCalculator,
      leftCalc: ImpurityCalculator,
      rightCalc: ImpurityCalculator,
      metadata: DecisionTreeMetadata): ImpurityStats = {

    val impurity: Double = fullCalc.calculate()

    val leftCount = leftCalc.count
    val rightCount = rightCalc.count
    val totalCount = leftCount + rightCount

    // If left child or right child doesn't satisfy minimum instances per node,
    // then this split is invalid, return invalid information gain stats.
    if ((leftCount < metadata.minInstancesPerNode) ||
      (rightCount < metadata.minInstancesPerNode)) {
      return ImpurityStats.getInvalidImpurityStats(fullCalc)
    }

    val leftImpurity = leftCalc.calculate() // Note: This equals 0 if count = 0
    val rightImpurity = rightCalc.calculate()

    val leftWeight = leftCount / totalCount.toDouble
    val rightWeight = rightCount / totalCount.toDouble
    val gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity

    // if information gain doesn't satisfy minimum information gain,
    // then this split is invalid, return invalid information gain stats.
    if (gain < metadata.minInfoGain) {
      return ImpurityStats.getInvalidImpurityStats(fullCalc)
    }

    new ImpurityStats(gain, impurity, fullCalc, leftCalc, rightCalc)
  }

  /**
   * Find the best split for an ordered categorical feature at a single node.
   *
   * Algorithm:
   *  - For each category, compute a "centroid."
   *     - For multiclass classification, the centroid is the label impurity.
   *     - For binary classification and regression, the centroid is the average label.
   *  - Sort the centroids, and consider splits anywhere in this order.
   *    Thus, with K categories, we consider K - 1 possible splits.
   *
   * @param featureIndex  Index of feature being split.
   * @param values  Feature values at this node.  Sorted in increasing order.
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseOrderedCategoricalSplit(
      featureIndex: Int,
      values: Array[Int],
      indices: Array[Int],
      labels: Array[Double],
      from: Int,
      to: Int,
      featureArity: Int,
      metadata: DecisionTreeMetadata,
      fullImpurityAgg: ImpurityAggregatorSingle): (Option[Split], ImpurityStats) = {
    // TODO: Support high-arity features by using a single array to hold the stats.
    // aggStats(category) = label statistics for category
    val aggStats = getAggStats(metadata, featureIndex, from, to, values, labels, indices)

    // Compute centroids.  centroidsForCategories is a sequence of pairs: (category, centroid)
    val centroidsForCategories = 0.until(featureArity).map { cat =>
      (cat, getCentroid(metadata, aggStats(cat)))
    }

    val categoriesSortedByCentroid: List[Int] = centroidsForCategories.toList.sortBy(_._2).map(_._1)

    // Cumulative sums of bin statistics for left, right parts of split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = metadata.createImpurityAggregator()
    aggStats.foreach(rightImpurityAgg.add(_))


    val fullImpurity = rightImpurityAgg.getCalculator.calculate()

    // Consider all splits. These only cover valid splits, with at least one category on each side.
    val numSplits = categoriesSortedByCentroid.length - 1

    // TODO(smurching): Just compute best split & its stats outright in this section?
    // Compute the index & gain of the best split, along with an impurity aggregator containing
    // stats for the left side of the best split.
    // Initial values: bestSplitIdx = -1, bestGain = 0.0, bestImpurityAgg = an empty impurity
    // aggregator
    val (bestSplitIndex, bestGain, bestLeftAgg)
    = 0.until(numSplits).foldLeft((-1, 0.0, leftImpurityAgg.deepCopy())) {
      case ((splitIdx, gain, currBestLeftAgg), sortedCatIndex) =>
        val cat = categoriesSortedByCentroid(sortedCatIndex)
        // Update left, right stats
        val catStats = aggStats(cat)
        leftImpurityAgg.add(catStats)
        rightImpurityAgg.subtract(catStats)

        val stats = calculateImpurityStats(fullImpurityAgg,
          leftImpurityAgg, rightImpurityAgg, metadata)

        if (stats.valid && stats.gain > gain) {
          (sortedCatIndex, stats.gain, leftImpurityAgg.deepCopy())
        } else {
          (splitIdx, gain, currBestLeftAgg)
        }
    }

    // Given the index of the best split, create a corresponding split object and return
    // the best split & its impurity statistics
    val categoriesForSplit =
      categoriesSortedByCentroid.slice(0, bestSplitIndex + 1).map(_.toDouble)
    val bestFeatureSplit =
      new CategoricalSplit(featureIndex, categoriesForSplit.toArray, featureArity)
    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftAgg.getCalculator, bestRightImpurityAgg.getCalculator)

    if (bestSplitIndex == -1 || bestGain == 0.0) {
      (None, bestImpurityStats)
    } else {
      (Some(bestFeatureSplit), bestImpurityStats)
    }

  }


  /**
   * Find the best split for an unordered categorical feature at a single node.
   *
   * Algorithm:
   *  - Considers all possible subsets (exponentially many)
   *
   * @param featureIndex  Index of feature being split.
   * @param values  Feature values at this node.  Sorted in increasing order.
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseUnorderedCategoricalSplit(
      featureIndex: Int,
      values: Array[Int],
      indices: Array[Int],
      labels: Array[Double],
      from: Int,
      to: Int,
      featureArity: Int,
      metadata: DecisionTreeMetadata,
      splits: Array[CategoricalSplit],
      fullImpurityAgg: ImpurityAggregatorSingle): (Option[Split], ImpurityStats) = {

    // Label stats for each category
    val aggStats = getAggStats(metadata, featureIndex, from, to, values, labels, indices)

    // Aggregated statistics for left part of split and entire split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val fullImpurity = fullImpurityAgg.getCalculator.calculate()

    if (featureArity == 1) {
      // All instances go right
      val impurityStats = new ImpurityStats(0.0, fullImpurityAgg.getCalculator.calculate(),
        fullImpurityAgg.getCalculator, leftImpurityAgg.getCalculator,
        fullImpurityAgg.getCalculator)
      (None, impurityStats)
    } else {
      //  TODO: We currently add and remove the stats for all categories for each split.
      //  A better way to do it would be to consider splits in an order such that each iteration
      //  only requires addition/removal of a single category and a single add/subtract to
      //  leftCount and rightCount.
      //  TODO: Use more efficient encoding such as gray codes
      var bestSplit: Option[CategoricalSplit] = None
      val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
      var bestGain: Double = -1.0
      val fullCount: Double = to - from
      for (split <- splits) {
        // Update left, right impurity stats
        split.leftCategories.foreach(c => leftImpurityAgg.add(aggStats(c.toInt)))
        val rightImpurityAgg = fullImpurityAgg.deepCopy().subtract(leftImpurityAgg)
        val leftCount = leftImpurityAgg.getCount
        val rightCount = rightImpurityAgg.getCount
        // Compute impurity
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
          bestSplit = Some(split)
          System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats,
            0, leftImpurityAgg.stats.length)
          bestGain = gain
        }
        // Reset left impurity stats
        leftImpurityAgg.clear()
      }

      val bestFeatureSplit = bestSplit match {
        case Some(split) => Some(
          new CategoricalSplit(featureIndex, split.leftCategories, featureArity))
        case None => None

      }
      val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
      val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity,
        fullImpurityAgg.getCalculator, bestLeftImpurityAgg.getCalculator,
        bestRightImpurityAgg.getCalculator)
      (bestFeatureSplit, bestImpurityStats)
    }
  }

  /**
   * Choose splitting rule: feature value <= threshold
   *
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats may still be
   *          useful, so they are returned.
   */
  private[impl] def chooseContinuousSplit(
      featureIndex: Int,
      values: Array[Int],
      indices: Array[Int],
      labels: Array[Double],
      from: Int,
      to: Int,
      metadata: DecisionTreeMetadata,
      splits: Array[Array[Split]],
      fullImpurityAgg: ImpurityAggregatorSingle): (Option[Split], ImpurityStats) = {

    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = fullImpurityAgg.deepCopy()

    var bestThreshold: Double = Double.NegativeInfinity
    val bestLeftImpurityAgg = metadata.createImpurityAggregator()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Int = 0
    var rightCount: Int = to - from
    val fullCount: Double = rightCount
    // Look up the threshold corresponding to the first binned value, or use -Infinity
    // if values is empty
    var currentThreshold: Double = values.headOption match {
      case None => bestThreshold
      case Some(binnedVal) =>
        splits(featureIndex)(binnedVal).asInstanceOf[ContinuousSplit].threshold
    }
    var j = from
    val numSplits = metadata.numBins(featureIndex) - 1
    while (j < to) {
      // Look up the least upper bound on the feature values in the current bin
      // TODO(smurching): When using continuous feature values (as opposed to binned values),
      // avoid this lookup
      val value = if (values(j) < numSplits) {
        splits(featureIndex)(values(j)).asInstanceOf[ContinuousSplit].threshold
      } else {
        Double.PositiveInfinity
      }
      val label = labels(indices(j))
      if (value != currentThreshold) {
        // Check gain
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
          bestThreshold = currentThreshold
          System.arraycopy(leftImpurityAgg.stats, 0,
            bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
          bestGain = gain
        }
        currentThreshold = value
      }
      // Move this instance from right to left side of split.
      leftImpurityAgg.update(label, 1)
      rightImpurityAgg.update(label, -1)
      leftCount += 1
      rightCount -= 1
      j += 1
    }

    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)
    val split = if (bestThreshold != Double.NegativeInfinity && bestThreshold != values.last) {
      Some(new ContinuousSplit(featureIndex, bestThreshold))
    } else {
      None
    }
    (split, bestImpurityStats)

  }
}

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

package org.apache.spark.ml.regression

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.OptimizedRandomForest
import org.apache.spark.ml.util._
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset


/**
  * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a>
  * learning algorithm for regression.
  * It supports both continuous and categorical features.
  */
@Since("1.4.0")
class OptimizedRandomForestRegressor @Since("1.4.0") (@Since("1.4.0") override val uid: String)
  extends Predictor[Vector, OptimizedRandomForestRegressor, RandomForestRegressionModel]
    with RandomForestRegressorParams with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("rfr"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeRegressorParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
    * Specifies how often to checkpoint the cached node IDs.
    * E.g. 10 means that the cache will get checkpointed every 10 iterations.
    * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
    * [[org.apache.spark.SparkContext]].
    * Must be at least 1.
    * (default = 10)
    * @group setParam
    */
  @Since("1.4.0")
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setImpurity(value: String): this.type = set(impurity, value)

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from RandomForestParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

  override protected def train(dataset: Dataset[_]): RandomForestRegressionModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(labelCol, featuresCol, predictionCol, impurity, numTrees,
      featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, cacheNodeIds, checkpointInterval)

    val trees = OptimizedRandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeRegressionModel])

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestRegressionModel(uid, trees, numFeatures)
    instr.logSuccess(m)
    m
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): OptimizedRandomForestRegressor = defaultCopy(extra)
}

@Since("1.4.0")
object OptimizedRandomForestRegressor extends DefaultParamsReadable[OptimizedRandomForestRegressor]{
  /** Accessor for supported impurity settings: variance */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = TreeRegressorParams.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  @Since("1.4.0")
  final val supportedFeatureSubsetStrategies: Array[String] =
  TreeEnsembleParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): OptimizedRandomForestRegressor = super.load(path)

}


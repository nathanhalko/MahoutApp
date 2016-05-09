package com.nhalko.mahoutapp

import scala.util.Try

import org.apache.spark._
import org.apache.spark.rdd.RDD
// Import classes for MLLib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object DecisionTrees {

  def main(args: Array[String]) {}

  def run()(implicit sc: SparkContext) = {

    // define the Flight Schema
    case class Flight(dofM: String, dofW: String, carrier: String, tailnum: String, flnum: Int, org_id: String,
                      origin: String, dest_id: String, dest: String, crsdeptime: Double, deptime: Double, depdelaymins: Double,
                      crsarrtime: Double, arrtime: Double, arrdelay: Double, crselapsedtime: Double, dist: Int)

    // function to parse input into Flight class
    def parseFlight(str: String): Flight = {
      val line = str.split(",")
      Flight(line(0), line(1), line(2), line(3), line(4).toInt, line(5), line(6), line(7), line(8),
        line(9).toDouble, line(10).toDouble, line(11).toDouble, line(12).toDouble, line(13).toDouble,
        line(14).toDouble, line(15).toDouble, line(16).toInt)
    }

    // parse the RDD of csv lines into an RDD of flight classes
    val linesCnt = sc.accumulator(0L, "total lines")
    val flightsRDD = sc.textFile("/Users/nhalko/Documents/NYC_2016/datasets/flights/rita2014jan.csv")
      .flatMap { line =>
        linesCnt += 1
        Try { parseFlight(line) }.toOption
      }
      .cache()

    // use .count to trigger the computation and tick the counters
    val validCnt = flightsRDD.count()
    val badCnt = linesCnt.value - validCnt
    println(s"$badCnt ${"%.4f".format(badCnt / linesCnt.value.toDouble * 100)}% bad lines out of ${linesCnt.value} total.")

    /**
      * Transform non-numeric data into numeric values
      */

    val carrierMap: Map[String, Int] = flightsRDD.map(_.carrier).distinct.collect()
      .zipWithIndex
      .toMap

    val originMap: Map[String, Int] = flightsRDD.map(_.origin).distinct.collect
      .zipWithIndex
      .toMap

    val destMap: Map[String, Int] = flightsRDD.map(_.dest).distinct.collect
      .zipWithIndex
      .toMap

    /**
      * Define the features array as a LabeledPoint.  Select only relevant features
      */
    val mldata = flightsRDD.map { flight =>
      LabeledPoint(
        if (flight.depdelaymins > 40) 1.0 else 0.0,
        Vectors.dense(
          flight.dofM.toDouble - 1,           // day of month
          flight.dofW.toDouble - 1,           // day of week
          flight.crsdeptime,                  // departure time
          flight.crsarrtime,                  // arrival time
          carrierMap(flight.carrier).toDouble,// carrier
          flight.crselapsedtime,              // elapsed flight time
          originMap(flight.origin).toDouble,  // departure city
          destMap(flight.dest).toDouble       // arrival city
        )
      )
    }.cache()

    /**
      * Partition the data into training and test data sets
      */
    // mldata0 is %85 not delayed flights
    val mldata0 = mldata.filter(x => x.label == 0).randomSplit(Array(0.85, 0.15))(1)
    // mldata1 is %100 delayed flights
    val mldata1 = mldata.filter(x => x.label != 0)
    // mldata2 is delayed and not delayed
    val mldata2 = mldata0 ++ mldata1

    //  split mldata2 into training and test data
    val splits = mldata2.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    /**
      *  Define some parameters and train the model.
      */

    // airity of non-continuous features
    val categoricalFeaturesInfo = Map(
      0 -> 31,
      1 -> 7,
      4 -> carrierMap.size,
      6 -> originMap.size,
      7 -> destMap.size
    )

    val numClasses = 2      // 0,1
    val impurity   = "gini" // or 'entropy'
    val maxDepth   = 3 //9      // depth of the decision tree
    val maxBins    = 500 //7000   // discretization of continuous features

    // call DecisionTree trainClassifier with the trainingData , which returns the model
    val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // print out the decision tree
    println(model.toDebugString)

    /**
      * Evaluate model on test instances and compute test error
      */

    val obsCnt = sc.accumulator(0.0, "observation count")
    val wrongPred = testData.filter { point =>
      obsCnt += 1
      point.label != model.predict(point.features)
    }.count()

    val ratioWrong = wrongPred / obsCnt.value
    println(s"The model is wrong ${"%.4f".format(ratioWrong * 100)}% of the time.")
  }
}
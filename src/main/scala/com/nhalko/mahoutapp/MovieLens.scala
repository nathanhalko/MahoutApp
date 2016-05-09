package com.nhalko.mahoutapp

import java.io.File
import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}


object MovieLens {

  def main(args: Array[String]): Unit = {}

  def run()(implicit sc: SparkContext) = {

    /**
      * Load the data into Spark
      */
    val movieLensHomeDir = "/Users/nhalko/Documents/NYC_2016/datasets/movielens"

    val movies = sc.textFile(s"${movieLensHomeDir}/movies.dat").map { line =>
      // MovieID::Title::Genres
      // 1::Toy Story (1995)::Animation|Children's|Comedy
      val Array(id, name, _) = line.split("::", 3)
      id.toInt -> name
    }.collect.toMap

    val ratings = sc.textFile(s"${movieLensHomeDir}/ratings.dat").map { line =>
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      // UserID::MovieID::Rating::Timestamp
      // 1::1193::5::978300760
      val Array(userId, movieId, rating, timeStamp) = line.split("::", 4)

      (timeStamp.toLong % 10, Rating(userId.toInt, movieId.toInt, rating.toDouble))
    }

    /**
      * Partition into Training 60%, Validation 20%, Test 20%
      */

    val training   = ratings.filter{case (ts, _) => ts < 6}.values.cache()
    val validation = ratings.filter{case (ts, _) => ts >= 6 && ts < 8}.values.cache()
    val test       = ratings.filter{case (ts, _) => ts >= 8}.values.cache()

//    println(s"""%table DataSet\tCount
//Training\t${training.count()}
//Validation\t${validation.count()}
//Test\t${test.count()}""")

    /**
      * Define the RMSE for evaluating model performance
      */
    def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {

      val predictions: RDD[Rating] = model.predict( data.map {r => r.user -> r.product} )

      val predictionsAndRatings = predictions
        .map {p => (p.user, p.product) -> p.rating}
        .join(data.map {r => (r.user, r.product) -> r.rating})
        .values

      math.sqrt(
        predictionsAndRatings
          .map{ case (pred, actual) => math.pow(pred - actual, 2) }
          .mean
      )
    }

    /**
      * Train the model, find the best one and evaluate performance
      */

    case class ModelResult(
                            model: Option[MatrixFactorizationModel] = None,
                            rmse: Double = Double.MaxValue,
                            rank: Int =  0,
                            lambda: Double = -1.0,
                            numIter: Int = -1
                          )


    val modelResults = for {
      rank <- List(8, 12)
      lambda <- List(0.1, 10.0)
      numIter <- List(10) // List(10, 20)
    } yield {
        val model = ALS.train(training, rank, numIter, lambda)
        val validationRmse = computeRmse(model, validation)

        ModelResult(Some(model), validationRmse, rank, lambda, numIter)
      }

    // find the model with minimum error
    val best = modelResults.minBy(_.rmse)
    val bestModel = best.model.get
    // display results
//    println(s"""%table RMSE (validation)\trank\tlambda\tnumIters
//${modelResults.map(mr => s"${mr.rmse}\t${mr.rank}\t${mr.lambda}\t${mr.numIter}").mkString("\n")}""")

    // test it on the test data
    val testRmse = computeRmse(bestModel, test)
    println(s"The best model was trained with rank = ${best.rank} and lambda = ${best.lambda} and numIter = ${best.numIter} and its RMSE on the test set is $testRmse")

    // create a naive baseline and compare it with the best model
    val meanRating   = training.union(validation).map(_.rating).mean
    val baselineRmse = math.sqrt(test.map {r => math.pow(meanRating - r.rating, 2)}.mean)
    val improvement  = (baselineRmse - testRmse) / baselineRmse * 100

    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    // save for use in making recommendations (output dir must not exist)
    //bestModel.save(sc, s"$movieLensHomeDir/model")

    /**
      * Create a method to make the recommendations
      */
    val moviesWithGenres = sc.textFile(s"${movieLensHomeDir}/movies.dat").map { line =>
      // MovieID::Title::Genres
      // 1::Toy Story (1995)::Animation|Children's|Comedy
      val Array(id, _, genres) = line.split("::", 3)
      id.toInt -> genres
    }.collect.toMap


    def recommend(userId: Int, genre: String = "") = {
      val candidates = sc.parallelize(
        moviesWithGenres.filter {case (id, genres) => genres.toLowerCase.contains(genre.toLowerCase)}.keys.toSeq
      )

      val userRatings = ratings
        .filter {case (_, r) => r.user == userId}
        .map {case (_, r) => movies(r.product) -> r.rating}
        .collect()

      val recs =  bestModel
        .predict(candidates.map {movieId => userId -> movieId})
        .collect()
        .sortBy(- _.rating)
        .take(5)
        .map {r =>
          movies(r.product)
        }

      (recs, userRatings)
    }

    /**
      * Make some recommendations
      */


//    val genres = moviesWithGenres.values.toList.flatMap(_.split("\\|")).distinct.map(g => (g,g))
//    val genre  = z.select("genre", genres)
//    val userId = z.input("userId", "100").asInstanceOf[String]
//
//    val (recs, userRatings) = recommend(userId.toInt, genre.asInstanceOf[String])
//    println(s"\n\n\nRecommended for user: \n${recs.mkString("\n")}")
//    println(s"\n\nBased on:\n${userRatings.mkString("\n")}")

  }
}
package com.nhalko.mahoutapp


import scala.util.Try

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg._


object ClusteringTheNews {

  def main(args: Array[String]): Unit = {

  }

  def run()(implicit sc: SparkContext) = {
    /**
      * Define some helper functions for later use
      */

    // vector addition
    def sumArray(m: Array[Double], n: Array[Double]): Array[Double] = m.zip(n).map {case (i,j) => i + j}

    // scalar multiplication 1/a
    def divArray(m: Array[Double], divisor: Double) : Array[Double] = m.map(_ / divisor)

    def wordToVector(w: String, m: Word2VecModel): Vector = {
      Try {
        m.transform(w)
      } getOrElse {
        println(s"$w failed to transform")
        Vectors.zeros(100)
      }
    }

    /**
      * ... Seq(President, of, China, lunches, with, Brazilian, President)
      *     Seq(Palestinians, to, elect, new, president, on, January, 9) ...
      */
    val news_titles: RDD[Seq[String]] = sc.textFile("/Users/nhalko/Documents/NYC_2016/datasets/wikinews/wikinews_titles.txt")
      .map(_.split(" ").toSeq)
      .cache()

    /**
      * ... Seq(President)
      *     Seq(of)
      *     Seq(China) ...
      */
    val news_titles_words: RDD[Seq[String]] = news_titles.flatMap {
      words =>
        words.map(word => Seq(word))
    }.cache()

    // fit a model that can compute synonyms
    val word2vec = new Word2Vec()
    val model = word2vec.fit(news_titles_words)

    /**
      * Create a title vector from each word vector by taking
      * an average.  So the title vector is the average vector of all
      * its words.
      */
    val title_vectors = news_titles.map { title =>
      val tVec = title
        .map(word => wordToVector(word, model).toArray)
        .reduceLeft(sumArray)
        .map(_ / title.length)

      new DenseVector(tVec).asInstanceOf[Vector]
    }.cache()

    // Make a tuple of (actualTitle, title_vector)
    val title_pairs = news_titles.zip(title_vectors)

    val numClusters = 100
    val numIterations = 25

    val clusters = KMeans.train(title_vectors, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val wssse = clusters.computeCost(title_vectors)

    // predict the topic for each title
    val article_membership: RDD[(Int, String)] = title_pairs.map { case (titleSeq, titleVector) =>
      // topicId -> title String
      clusters.predict(titleVector) -> titleSeq.mkString(" ")
    }.cache()

    // number each cluster center
    val cluster_centers: RDD[(Int, Vector)] = sc.parallelize(
      clusters.clusterCenters.zipWithIndex.map {case (center, idx) => idx -> center}
    )

    // synonyms are an Array of length 5 with (word, distance) tuples
    // Map away the 'distance' and concatenate to form the cluster topic
    val cluster_topics: RDD[(Int, String)] = cluster_centers.map {case (idx, center) =>
      idx -> model.findSynonyms(center, 5)
        .sortBy {case (word, dist) => dist}  // sort by the distance to put the best fits first
        .map {case (word, dist) => word}
        .mkString(" ")
    }
    println(s"${cluster_topics.take(3).mkString("\n")}")

    val sample_topics = cluster_topics.collect
    def sample_members(topicId: Int) = {
      article_membership
        .filter {case (id, title) => id == topicId}
        .map {case (_, title) => title}
        .take(25)
    }

    //    val topic = z.select("topic", sample_topics.map {case (id, title) => id.toString -> title})

    //    println(s"%table Related Articles:\n${sample_members(topic.toString.toInt).mkString("\n")}")

    val topicMap = sample_topics.toMap

    def topicMe(title: String) = {
      val tVec = {
        val tSeq = title.split(" ")
        val tArr = tSeq
          .map(word => wordToVector(word, model).toArray)
          .reduceLeft(sumArray)
          .map(_ / tSeq.length)

        new DenseVector(tArr).asInstanceOf[Vector]
      }

      val topicId = clusters.predict(tVec)

      topicMap(topicId)
    }

    //    println(s"\n\nBelongs to topic: ${topicMe(z.input("Type some words:", "Here comes the sun").toString)}")
  }
}
package com.nhalko.mahoutapp

import scala.util.Random

import org.scalatest.{Matchers, FunSuite}
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite

import org.apache.log4j.Level

import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._

import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.RLikeOps._

/**
  * Created by nhalko on 4/18/16.
  */
class RidgeRegression extends FunSuite with DistributedSparkSuite with Matchers {

  private final implicit val log = getLog(classOf[RidgeRegression])

  /**
    * Distributed ridge
    */
  def dridge(drmX: DrmLike[Int], y: Vector, lambda: Double): Vector = {
    require(drmX.nrow == y.length, "Target and dataset have different point count.")

    // Add bias term
    val drmXB = (1.0 cbind drmX).checkpoint()

    // A = X'X + lambda*I
    val mxA = drmXB.t %*% drmXB
    mxA.diagv += lambda

    // b = X'y
    val b = (drmXB.t %*% y).collect(::, 0)

    // Solve A*beta=b for beta
    val beta = solve(mxA, b)

    // return solution
    beta
  }

  /**
    * Create simulated data: Given a solution vector `beta` return a system such that
    * beta is a close solution to A*beta=y
    */
  def simData(beta: Vector, m: Int, noiseSigma: Double = 0.04): (Matrix, Vector) = {
    val n = beta.length
    val mxData = Matrices.symmetricUniformView(m, n, 1234).cloned

    // Bias always 1
    mxData(::, 0) = 1
    val y = mxData %*% beta

    // perturb y with a little noise
    y := {
      v =>
        v + noiseSigma * Random.nextGaussian()
    }

    // Return simulated X and y
    mxData(::, 1 until n) -> y
  }

  /**
    * ACTION!
    */
  test("ols") {
    setLogLevel(Level.TRACE)

    // Simulated beta
    val betaSim = dvec(3, 25, 10, -4)

    // Simulated data with noise
    val (mxX, y) = simData(betaSim, 250)

    // Run distributed ridge
    val drmX = drmParallelize(mxX, numPartitions = 2)
    trace(s"X dim: ${drmX.nrow} x ${drmX.ncol}, y dim: ${y.length}")
    val fittedBeta = dridge(drmX, y, 0)
    trace(s"fittedBeta:$fittedBeta")

    val error = (betaSim - fittedBeta).norm(1)
    trace(s"error: $error")
    error should be < 1e-1
  }
}


val mahoutVer = "0.12.0"
val sparkVer  = "1.5.2"


lazy val root = (project in file(".")).
  settings(
    name := "mahoutapp",
    version := "0.1",
    scalaVersion := "2.10.4",

    libraryDependencies ++= Seq(
      "org.apache.mahout" %% "mahout-math-scala" % mahoutVer,
      "org.apache.mahout" %% "mahout-spark" % mahoutVer,
      //      "org.apache.spark" %% "spark-core" % sparkVer

      "org.scalatest" %% "scalatest" % "2.2.4" % "test",
      "org.apache.mahout" %% "mahout-math-scala" % mahoutVer % "test" classifier "tests",
      "org.apache.mahout" %% "mahout-spark" % mahoutVer % "test" classifier "tests"
    )
  )

package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import made.WithSpark
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec._
import org.scalatest.matchers._

class MyLinearResressionSpec extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001

  import spark.implicits._

  "Model" should "fit parameters by data" in {
    val X = DenseMatrix.rand(100000, 3)
    val y = X * DenseVector(0.5, -0.1, 0.2)
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "y")

    val pipeline = new Pipeline()
      .setStages(
        Array(
          new VectorAssembler()
            .setInputCols(Array("x1", "x2", "x3"))
            .setOutputCol("features"),
          new MyLinearRegression()
            .setLabelCol("y")
        )
      )
    val model = pipeline.fit(df)
    val results = model.stages.last.asInstanceOf[MyLinearRegressionModel].coefficients
    results(0) should be (0.5 +- delta)
    results(1) should be (-0.1 +- delta)
    results(2) should be (0.2 +- delta)
  }

  "Estimator" should "should produce functional model" in {
    val train = spark.createDataFrame(Seq(
      LabeledPoint(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      LabeledPoint(0.0, Vectors.dense(3.0, 2.0, -0.1)),
      LabeledPoint(1.0, Vectors.dense(0.0, 2.2, -1.5))))

    val estimator = new MyLinearRegression()
      .setMaxIter(2)
    val model = estimator.fit(train.toDF())
    val test = spark.createDataFrame(Seq(
      LabeledPoint(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      LabeledPoint(0.0, Vectors.dense(3.0, 2.0, -0.1)),
      LabeledPoint(1.0, Vectors.dense(0.0, 2.2, -1.5))))
    model.transform(test)
  }

  "Estimator" should "work after re-read" in {

    val X = DenseMatrix.rand(100000, 3)
    val y = X * DenseVector(0.5, -0.1, 0.2)
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "y")

    val pipeline = new Pipeline()
      .setStages(
        Array(
          new VectorAssembler()
            .setInputCols(Array("x1", "x2", "x3"))
            .setOutputCol("features"),
          new MyLinearRegression()
            .setLabelCol("y")
        )
      )
    val model = pipeline.fit(df)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
    reRead.transform(df)

    val results = model.stages.last.asInstanceOf[MyLinearRegressionModel].coefficients
    results(0) should be (0.5 +- delta)
    results(1) should be (-0.1 +- delta)
    results(2) should be (0.2 +- delta)
  }
}

package org.apache.spark.ml.made

import breeze.linalg._
import org.apache.spark.ml.{Pipeline, PredictorParams}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}


object DeveloperApiExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("DeveloperApiExample")
      .config("spark.master", "local")
      .getOrCreate()
    import spark.implicits._

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
    println(model.stages.last.asInstanceOf[MyLinearRegressionModel].coefficients)

    spark.stop()
  }
}


private trait MyLinearRegressionParams extends PredictorParams {

  val maxIter: IntParam = new IntParam(this, "maxIter", "max number of iterations")
  def getMaxIter: Int = $(maxIter)

  val learningRate: DoubleParam = new DoubleParam(
    this, "learningRate", "gradient multiplier for GD step"
  )
  def getLearningRate: Double = $(learningRate)

}

private class MyLinearRegression(override val uid: String)
  extends Regressor[Vector, MyLinearRegression, MyLinearRegressionModel]
    with MyLinearRegressionParams {

  def this() = this(Identifiable.randomUID("myLogReg"))

  setMaxIter(1000)
  setLearningRate(0.1)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setLearningRate(lr: Double): this.type = set(learningRate, lr)


  override protected def train(dataset: Dataset[_]): MyLinearRegressionModel = {

    val data = extractLabeledPoints(dataset).persist()
    val dataPoints = data.count()

    val numFeatures = data.take(1)(0).features.size
    var coefficients = Vectors.zeros(numFeatures)
    val targets = data.map { _.getLabel }.persist()

    val lr = getLearningRate
    for (_ <- 0 until getMaxIter) {
      val predictions = data map {
        _.getFeatures.asBreeze dot coefficients.asBreeze
      }
      val differences = (predictions zip targets) map {
        case (p, t) => (p - t)
      }
      val stochGrads = (differences zip data) map {
        case (diff, datapoint) => diff * datapoint.getFeatures.asBreeze
      }
      val grad = stochGrads
        .treeReduce(_ + _)
        .map(_ / dataPoints)
      coefficients = Vectors.fromBreeze(coefficients.asBreeze -:- (lr * grad))

    }

    new MyLinearRegressionModel(uid, coefficients).setParent(this)
  }

  override def copy(extra: ParamMap): MyLinearRegression = defaultCopy(extra)
}


private class MyLinearRegressionModel(
                                       override val uid: String,
                                       val coefficients: Vector)
  extends RegressionModel[Vector, MyLinearRegressionModel]
    with MyLinearRegressionParams {


  override val numFeatures: Int = coefficients.size

  override def copy(extra: ParamMap): MyLinearRegressionModel = {
    copyValues(new MyLinearRegressionModel(uid, coefficients), extra).setParent(parent)
  }

  override def predict(features: Vector): Double = {
    BLAS.dot(features, coefficients)
  }
}

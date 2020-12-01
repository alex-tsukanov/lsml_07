package org.apache.spark.ml.made

import breeze.linalg._
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}


trait MyLinearRegressionParams extends PredictorParams {

  val maxIter: IntParam = new IntParam(this, "maxIter", "max number of iterations")
  def getMaxIter: Int = $(maxIter)

  val learningRate: DoubleParam = new DoubleParam(
    this, "learningRate", "gradient multiplier for GD step"
  )
  def getLearningRate: Double = $(learningRate)

}

class MyLinearRegression(override val uid: String)
  extends Regressor[Vector, MyLinearRegression, MyLinearRegressionModel]
    with MyLinearRegressionParams
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("my Linear Regression"))

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

object MyLinearRegression extends DefaultParamsReadable[MyLinearRegression]


class MyLinearRegressionModel(
                                       override val uid: String,
                                       val coefficients: Vector)
  extends RegressionModel[Vector, MyLinearRegressionModel]
    with MyLinearRegressionParams
    with MLWritable {

  private[made] def this(coefficients: Vector) = this(Identifiable.randomUID("linearRegression"), coefficients)

  override val numFeatures: Int = coefficients.size

  override def copy(extra: ParamMap): MyLinearRegressionModel = {
    copyValues(new MyLinearRegressionModel(uid, coefficients), extra).setParent(parent)
  }

  override def predict(features: Vector): Double = {
    BLAS.dot(features, coefficients)
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val rdd = sparkSession.sparkContext.parallelize(coefficients.toArray).map(a => Row(a))
      val schema = StructType(Array(StructField("coefficient", DoubleType, nullable = false)))
      sparkSession.createDataFrame(rdd, schema).write.parquet(path + "/vectors")
    }
  }
}

object MyLinearRegressionModel extends MLReadable[MyLinearRegressionModel] {
  override def read: MLReader[MyLinearRegressionModel] = new MLReader[MyLinearRegressionModel] {
    override def load(path: String): MyLinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val coefficients = vectors.select("coefficient").rdd.map(r => r(0).asInstanceOf[Double]).collect()

      new MyLinearRegressionModel(Vectors.dense(coefficients))
    }
  }
}


import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.fm.{FactorizationMachines, FactorizationMachinesModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.optim.configuration._
import org.apache.spark.ml.optim._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

/**
  * Created by niejiabin on 2018/7/3.
  */
object testFM {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("FactorizationMachinesExample")
      .master("local[*]")
      .getOrCreate()
    val train = spark.read.format("libsvm").load("data/a9a.tr")
    val test = spark.read.format("libsvm").load("data/a9a.te")

    val trainer = new FactorizationMachines()
      .setAlgo(Algo.fromString("binary classification"))
      .setSolver(Solver.fromString("pftrl"))
      .setDim((1, 1, 16))
      .setReParamsL1((1, 1, 1))
      .setRegParamsL2((0.1, 0.1,0.1))
      .setAlpha((0.1, 0.1, 0.1))
      .setBeta((1.0, 1.0, 1.0))
      .setInitStdev(0.01)
      // .setStepSize(0.1)
      .setTol(0.001)
      .setMaxIter(3)
      .setThreshold(0.5)
      // .setMiniBatchFraction(0.5)
      .setNumPartitions(4)


    val model: FactorizationMachinesModel = trainer.fit(train)
//    println(model.weights.size)
    println(model.weights)
//    val bias = model.weights(model.weights.size - 1)
//    val linerWeight = model.weights.toArray.slice(model.dim._3 * model.numFeatures, model.dim._3 * model.numFeatures + model.numFeatures).size

//    println(linerWeight)
    //    model.
    val trainResult = model.transform(train)

    trainResult.show(5,false)
    val trainpredictionAndLabels = trainResult.select("prediction", "label").rdd.map {
      case Row(probability: Double, label: Double) =>
        (probability, label)
    }


    val trainevaluator = new BinaryClassificationMetrics(trainpredictionAndLabels).areaUnderROC()

    val result = model.transform(test)

    result.show(5, false)
    val predictionAndLabels = result.select("prediction", "label").rdd.map {
      case Row(probability: Double, label: Double) =>
        (probability, label)
    }


    val evaluator = new BinaryClassificationMetrics(predictionAndLabels).areaUnderROC()
    println(s"trainauc : $trainevaluator  testauc $evaluator")



    //
    //    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    //    println("Accuracy: " + evaluator.evaluate(predictionAndLabel))
    spark.stop()
  }
}

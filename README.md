# Spark-FM
Factorization Machines is a general predictor like SVMs but is also able to estimate reliable parameters under very high sparsity. However, they are costly to scale to large amounts of data and large numbers of features. Spark-FM is a parallel implementation of factorization machines based on Spark. It aims to utilize Spark's in-memory computing to address above problems.

# Highlight
In order to meet users' demands, Spark-FM supports various of optimization methods to train the model as follows.
 + Mini-batch Stochastic Gradient Descent (MLlib)
 + L-BFGS (MLlib)
 + Parallel Stochastic Gradient Descent ([spark-optim](https://github.com/hibayesian/spark-optim))
 + Parallel Ftrl ([spark-optim](https://github.com/hibayesian/spark-optim))


# Examples
## Scala API
```scala
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
  .setDim((1, 1, 8))
  .setReParamsL1((0.1, 0.1, 0.1))
  .setRegParamsL2((0.01, 0.01, 0.01))
  .setAlpha((0.1, 0.1, 0.1))
  .setBeta((1.0, 1.0, 1.0))
  .setInitStdev(0.01)
  // .setStepSize(0.1)
  .setTol(0.001)
  .setMaxIter(1)
  .setThreshold(0.5)
  // .setMiniBatchFraction(0.5)
  .setNumPartitions(4)

val model = trainer.fit(train)
val result = model.transform(test)
 val trainpredictionAndLabels = trainResult.select("prediction", "label").rdd.map {
      case Row(probability: Double, label: Double) =>
        (probability, label)
    }
	 val evaluator = new BinaryClassificationMetrics(trainpredictionAndLabels).areaUnderROC()
	 
	 println(evaluator)
spark.stop()
```

# Requirements
Spark-FM is built against Spark 2.1.1.

# Build From Source
```scala
sbt package
```

# Licenses
Spark-FM is available under Apache Licenses 2.0.



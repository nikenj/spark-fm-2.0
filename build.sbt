name := "spark-fm"

version := "1.0"

scalaVersion := "2.11.8"

spName := "spark-fm"

sparkVersion := "2.2.1"

sparkComponents += "mllib"

//libraryDependencies ++= Seq(
//  "org.apache.spark" % "spark-core_2.11" % "2.2.1" %"provided",
//  "org.apache.spark" % "spark-hive_2.11" % "2.2.1" %"provided",
//  "org.apache.spark" % "spark-mllib_2.11" % "2.2.1" % "provided",
//  "joda-time" % "joda-time" % "2.9"
//)

resolvers += Resolver.sonatypeRepo("public")

spShortDescription := "spark-fm"

spDescription := """A parallel implementation of factorization machines based on Spark"""
  .stripMargin

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")
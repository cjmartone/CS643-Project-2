package com.amazonaws.samples;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SparkApplicaiton {

	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder().appName("Simple Application").config("spark.master", "local").getOrCreate();
	
		// Load the trained model
		PipelineModel model = PipelineModel.load("DecisionTreeModel");
	
		Dataset<Row> data = spark
				.read()
				.format("csv")
				.option("header", true)
				.option("inferSchema", true)
				.load("ValidationDataset.csv");
	
		// Index the quality column and fit onto the data set
		StringIndexerModel labelIndexer = new StringIndexer()
				.setInputCol("quality")
				.setOutputCol("indexedLabel")
				.fit(data);
		
		// Assemble all feature columns to be used later
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] {"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"})
				.setOutputCol("features");
		data = assembler.transform(data);
		
		// Index the features and fit onto the data set
		VectorIndexerModel featureIndexer = new VectorIndexer()
				.setInputCol("features")
				.setOutputCol("indexedFeatures")
				.fit(data);
		

		// Make predictions.
		Dataset<Row> predictions = model.transform(data);

		// Select example rows to display.
		predictions.select("predictedLabel", "quality", "features").show(5);

		// Select (prediction, true label) and compute test error.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setLabelCol("indexedLabel")
		  .setPredictionCol("prediction")
		  .setMetricName("accuracy");
		double accuracy = evaluator.evaluate(predictions);
		System.out.println("Accuracy = " + accuracy);
		MulticlassMetrics metrics = evaluator.getMetrics(predictions);
		System.out.println("F1 score = " + metrics.fMeasure(1));
	}
}

package com.amazonaws.samples;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public final class ModelTraining {

	public static void main(String[] args) throws Exception {

		SparkSession spark = SparkSession.builder().appName("Simple Application").config("spark.master", "local").getOrCreate();
		
		// Load the training data and validation data from the files
		Dataset<Row> trainingData = spark
				.read()
				.format("csv")
				.option("header", true)
				.option("inferSchema", true)
				.load("TrainingDataset.csv");

		// Index the quality column and fit onto the data set
		StringIndexerModel labelIndexer = new StringIndexer()
				.setInputCol("quality")
				.setOutputCol("indexedLabel")
				.fit(trainingData);

		// Assemble all feature columns to be used later
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] {"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"})
				.setOutputCol("features");
		trainingData = assembler.transform(trainingData);

		// Index the features and fit onto the data set
		VectorIndexerModel featureIndexer = new VectorIndexer()
				.setInputCol("features")
				.setOutputCol("indexedFeatures")
				.fit(trainingData);

		// Train the DecisionTree model.
		DecisionTreeClassifier dt = new DecisionTreeClassifier()
				.setLabelCol("indexedLabel")
				.setFeaturesCol("indexedFeatures");

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString()
				.setInputCol("prediction")
				.setOutputCol("predictedLabel")
				.setLabels(labelIndexer.labelsArray()[0]);

		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});
		
		PipelineModel model = pipeline.fit(trainingData);

		model.write().overwrite().save("DecisionTreeModel");
	}
}
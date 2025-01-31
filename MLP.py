import os
# os.system('clear')
import argparse
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, struct
from pyspark.sql.types import IntegerType, FloatType, StructType, StructField

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims):
        super(MLPClassifier, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes

def create_mlp_classifier(input_dim, num_classes, hidden_dims):
    model = MLPClassifier(input_dim, num_classes, hidden_dims)
    return model

def numpy_to_torch(numpy_array):
    return torch.from_numpy(numpy_array).float()

@pandas_udf(IntegerType())
def MLPClassifier_udf(batch_inputs):
    input_tensor = numpy_to_torch(batch_inputs.to_numpy())
    
    input_dim = input_tensor.shape[1]
    num_classes = 10
    hidden_dims = [1024 * 50]
    
    model = create_mlp_classifier(input_dim, num_classes, hidden_dims)
    
    with torch.no_grad():
        predictions = model(input_tensor)
    
    return pd.Series(predictions.numpy())

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="MLP Distributed Classification with PySpark")
    parser.add_argument('--n_input', type=int, default=10000, help="Number of sentences")
    parser.add_argument('--hidden_dim', type=int, default=1024, help="hidden_dim")
    parser.add_argument('--hidden_layer', type=int, default=50, help="hidden_layer")
    args = parser.parse_args()

    input_dim = 128
    num_classes = 10  
    hidden_dims = [args.hidden_dim * args.hidden_layer]

    x = torch.randn(args.n_input, input_dim)

    spark = SparkSession.builder \
        .appName("MLPDistributedInference") \
        .getOrCreate()

    try:
        df = pd.DataFrame(x.numpy())
        schema = StructType([StructField(f"col_{i}", FloatType(), True) for i in range(input_dim)])
        spark_df = spark.createDataFrame(df, schema=schema)

        start_time_1 = time.time()
        
        columns = [col(f"col_{i}") for i in range(input_dim)]
        result_df = spark_df.select(MLPClassifier_udf(struct(*columns)).alias("predicted_class"))
        
        predictions = result_df.collect()
        
        end_time_1 = time.time()
        time_1 = end_time_1 - start_time_1

        mlp_model = MLPClassifier(input_dim, num_classes, hidden_dims)
    
        start_time_2 = time.time()
        output = mlp_model(x)
        end_time_2 = time.time()
        time_2 = end_time_2 - start_time_2

        print(f"Time cost for spark and non-spark version: [{time_1:.3f},  {time_2:.3f}] seconds")

    finally:
        spark.stop()
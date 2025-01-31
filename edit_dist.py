import os
from itertools import combinations
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType
import numpy as np
from functools import partial

def edit_distance(pair):
    str1, str2 = pair
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],    
                                   dp[i - 1][j],    
                                   dp[i - 1][j - 1])  

    return dp[m][n]

def process_batch(batch):
    return [edit_distance(pair) for pair in batch]

def compute_edit_distance_multiprocess(pairs, num_workers):
    batch_size = len(pairs) // num_workers
    if batch_size == 0:
        batch_size = 1
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = []
        for batch_result in tqdm(pool.imap(process_batch, batches), 
                               total=len(batches), 
                               desc="Processing batches"):
            results.extend(batch_result)
    
    return results

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--csv_dir', type=str, default='simple-wiki-unique-has-end-punct-sentences.csv', help="Directory of csv file")
    parser.add_argument('--num_sentences', type=int, default=1000, help="Number of sentences")
    args = parser.parse_args()

    num_workers = multiprocessing.cpu_count()
    print(f'number of available cpu cores: {num_workers}')
    
    text_data = pd.read_csv(args.csv_dir)['sentence']
    text_data = text_data[:args.num_sentences]
    pair_data = list(combinations(text_data, 2))

    print("Running Spark implementation...")
    start_time = time.time()
    
    spark = SparkSession.builder \
        .appName("EditDistance") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    @pandas_udf(IntegerType())
    def edit_distance_udf(str1, str2):
        return pd.Series([edit_distance((s1, s2)) for s1, s2 in zip(str1, str2)])

    pairs_df = pd.DataFrame(pair_data, columns=['string1', 'string2'])
    spark_df = spark.createDataFrame(pairs_df)

    result_df = spark_df.select(edit_distance_udf('string1', 'string2').alias('distance'))
    spark_distances = result_df.collect()
    
    spark_time = time.time() - start_time
    print(f"Time taken (Spark): {spark_time:.2f} seconds")
    
    spark.stop()

    print("Running Multi-process implementation...")
    start_time = time.time()
    multiprocess_distances = compute_edit_distance_multiprocess(pair_data, num_workers)
    multiprocess_time = time.time() - start_time
    print(f"Time taken (multi-process): {multiprocess_time:.3f} seconds")

    print("Running Single-process implementation...")
    start_time = time.time()
    distances = []
    for pair in tqdm(pair_data, ncols=100):
        distances.append(edit_distance(pair))
    for_loop_time = time.time() - start_time
    print(f"Time taken (for-loop): {for_loop_time:.3f} seconds")

    print("Sumary:")
    print(f"Number of sentence pairs processed: {len(pair_data)}")
    print(f"Sample distances (first 5):")
    print("Spark:", [d.distance for d in spark_distances[:5]])
    print("Multiprocess:", multiprocess_distances[:5])
    print("For-loop:", distances[:5])

    print(f"Time cost (Spark, multi-process, for-loop): [{spark_time:.3f}, {multiprocess_time:.3f}, {for_loop_time:.3f}]")
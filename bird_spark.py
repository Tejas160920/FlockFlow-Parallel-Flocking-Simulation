from pyspark.sql import SparkSession
import os
# os.system('clear')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle as pk
import argparse
from get_gif import *

def compute_speed(velocity):
    return np.linalg.norm(velocity)

def limit_speed(velocity, min_speed, max_speed):
    speed = compute_speed(velocity)
    
    if speed < 1e-10:
        return np.zeros_like(velocity)
    
    if speed < min_speed:
        velocity = velocity / speed * min_speed
    elif speed > max_speed:
        velocity = velocity / speed * max_speed
        
    return velocity

def update_lead_bird_position(t, lead_bird_speed, lead_bird_radius):
    angle = lead_bird_speed * t / lead_bird_radius
    x = lead_bird_radius * np.cos(angle)
    y = lead_bird_radius * np.sin(angle) * np.cos(angle)
    z = lead_bird_radius * (1 + 0.5 * np.sin(angle / 5))
    return np.array([x, y, z])

def compute_forces(bird_position, positions, min_distance, max_distance):
    distances = np.linalg.norm(positions - bird_position, axis=1)
    
    close_neighbors = positions[distances < min_distance]
    separation_force = np.sum(
        [(bird_position - neighbor) / max(np.linalg.norm(bird_position - neighbor)**2, 1e-5)
         for neighbor in close_neighbors], axis=0
    ) if len(close_neighbors) > 0 else np.zeros(3)
    
    return separation_force

def update_positions(positions, velocities, num_birds, min_speed, max_speed, min_distance, max_distance, time_step):
    updated_positions = positions.copy()
    updated_velocities = velocities.copy()
    
    for i in range(1, num_birds):
        forces = compute_forces(positions[i], positions, min_distance, max_distance)
        updated_velocities[i] += forces
        updated_velocities[i] = limit_speed(updated_velocities[i], min_speed, max_speed)
        updated_positions[i] += updated_velocities[i] * time_step
    
    return updated_positions, updated_velocities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bird Flocking Simulation with PySpark")
    parser.add_argument('--num_birds', type=int, default=1000, help="Number of birds in the simulation")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("BirdFlockingSimulation").getOrCreate()

    num_birds = args.num_birds
    num_frames = 500
    time_step = 1 / 4
    std_dev_position = 10.0
    lead_bird_speed = 20.0
    lead_bird_radius = 300.0
    min_speed = 10.0
    max_speed = 30.0
    min_distance = 10.0
    max_distance = 20.0

    positions = np.random.normal(loc=np.array([0, 0, 1.5 * lead_bird_radius]), scale=std_dev_position, size=(num_birds, 3))
    velocities = np.zeros((num_birds, 3))

    simulation = []
    time_cost = []

    for frame in range(num_frames):
        start_time = time.time()
        
        positions[0] = update_lead_bird_position(frame * time_step, lead_bird_speed, lead_bird_radius)

        rdd = spark.sparkContext.parallelize(range(1, num_birds))
        updated_data = rdd.map(
            lambda i: (i, positions[i], velocities[i])
        ).map(
            lambda data: (
                data[0],
                positions[data[0]] + limit_speed(
                    velocities[data[0]] + compute_forces(data[1], positions, min_distance, max_distance),
                    min_speed,
                    max_speed
                ) * time_step,
                limit_speed(
                    velocities[data[0]] + compute_forces(data[1], positions, min_distance, max_distance),
                    min_speed,
                    max_speed
                )
            )
        ).collect()

        for i, pos, vel in updated_data:
            positions[i] = pos
            velocities[i] = vel

        simulation.append(positions.copy())
        time_cost.append(time.time() - start_time)
        print(f'frame simulation time: {time_cost[-1]:.4f}s')

    mean_time = np.mean(time_cost)
    print(f"Average time cost per frame: {mean_time:.4f}")

    # create_compressed_gif(simulation, gif_name="bird_simulation.gif", duration=100, loop=1, resize_factor=0.5)

    spark.stop()

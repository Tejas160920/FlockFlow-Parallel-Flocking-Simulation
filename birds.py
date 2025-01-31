import os
os.system('clear')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle as pk
import argparse
from get_gif import *

# Function to compute the speed of a bird
def compute_speed(velocity):
    return np.linalg.norm(velocity)

# Function to limit speed within min and max range
def limit_speed(velocity, min_speed, max_speed):
    speed = compute_speed(velocity)
    
    # If speed is zero, return zero velocity to avoid division by zero
    if speed < 1e-10:  # Small threshold to prevent zero-division
        return np.zeros_like(velocity)
    
    # Scale the velocity to be within the min and max speed limits
    if speed < min_speed:
        velocity = velocity / speed * min_speed
    elif speed > max_speed:
        velocity = velocity / speed * max_speed
        
    return velocity


# Update the lead bird's position following a figure-eight (infinity) trajectory
def update_lead_bird_position(t):
    
    angle = lead_bird_speed * t / lead_bird_radius  # Control the speed of the bird's movement
    # Parametric equations for the figure-eight (infinity sign)
    x = lead_bird_radius * np.cos(angle)
    y = lead_bird_radius * np.sin(angle) * np.cos(angle)
    z = lead_bird_radius * (1 + 0.5 * np.sin(angle / 5))
    
    return np.array([x, y, z])

# Cohesion and separation functions combined
def compute_forces(bird_position, positions):
    distances = np.linalg.norm(positions - bird_position, axis=1)
    
    # Following leader
    d_lead = distances[0]
    lead_force = (positions[0] - bird_position) * ((1 / (d_lead))) if d_lead > 10 else np.zeros(3)

    # Following nearest neighbor when at lost
    nearest_idx = np.argmin(distances)
    d_near = distances[nearest_idx]
    cohesion_force = np.nan_to_num((positions[nearest_idx] - bird_position) * ((d_near / 1) ** 2)) if d_near > max_distance else np.zeros(3)
    
    # Separation to avoid close neighbors within min_distance
    close_neighbors = positions[distances < min_distance]
    close_distances = distances[distances < min_distance]
    separation_force = np.sum([(bird_position - neighbor) / (dist ** 2)
                         for neighbor, dist in zip(close_neighbors, close_distances) if dist > 0],
                        axis=0) if len(close_neighbors) > 0 else np.zeros(3)
    
    total_weight = np.sum([1 / ((dist / 1) ** 2) for dist in close_distances if dist > 0])
    if total_weight > 0:
        separation_force = separation_force / total_weight
    
    return cohesion_force + separation_force + lead_force

def update_positions(positions):

    # Update positions of other birds
    for i in range(1, num_birds):
        # Update velocity based on weighted forces
        velocities[i] += compute_forces(positions[i], positions)
        # Limit the bird's speed
        velocities[i] = limit_speed(velocities[i], min_speed, max_speed)
        # Update bird's position
        positions[i] += velocities[i] * time_step

    return positions

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--num_birds', type=int, default=10000, help="Number of birds")
    args = parser.parse_args()
    
    # Simulation parameters
    num_birds = args.num_birds
    num_frames = 500

    time_step = 1 / 4

    std_dev_position = 10.0
    lead_bird_speed = 20.0
    lead_bird_radius = 300.0
    min_speed = 10.0
    max_speed = 30.0
    max_distance = 20.0
    min_distance = 10.0

    # Initialize bird positions with Gaussian distribution
    positions = np.random.normal(loc=np.array([0, 0, 1.5*lead_bird_radius]), scale=std_dev_position, size=(num_birds, 3))
    velocities = np.zeros((num_birds, 3))


    simulation = []
    time_cost = []
    for frame in range(num_frames):

        start = time.time()

        # Update lead bird position and broadcast it
        positions[0] = update_lead_bird_position(frame * time_step)

        # update other bird position
        positions = update_positions(positions)
        
        end = time.time()
        frame_cost = end - start
        time_cost.append(frame_cost)

        simulation.append(positions.copy())

        print(f'frame simulation time: {frame_cost:.4f}s')
        
    mean_time = np.mean(time_cost)
    print(f'Average time cost per frame: {mean_time:.4f}')

    # # Visualization setup
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # save all frame 
    visualize_simulation(simulation, lead_bird_radius)

    # Usage example
    create_compressed_gif("./plot", gif_name="bird_simulation.gif", duration=100, loop=1, resize_factor=0.5)

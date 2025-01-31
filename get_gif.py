from PIL import Image
import os
# os.system('clear')
import pickle as pk
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image, ImageSequence
import os
import cv2

def create_mp4_from_images(folder_path, output_video="output.mp4", fps=10, resize_factor=0.6):
    # Get all image file paths in the folder, sorted by frame number
    images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))], 
                    key=lambda x: int(x.split('frame_')[1].split('.')[0]))

    # Read the first image to determine the video size
    first_image = Image.open(images[0])
    width, height = first_image.size
    width = int(width * resize_factor)
    height = int(height * resize_factor)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Process and add each frame to the video
    for img_path in images:
        img = Image.open(img_path)
        
        # Resize each frame based on the resize_factor
        img = img.resize((width, height), Image.ANTIALIAS)
        
        # Convert PIL image to a format compatible with OpenCV
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"MP4 video created and saved as {output_video}")

def create_compressed_gif(folder_path, gif_name="output.gif", duration=100, loop=1, resize_factor=0.6, optimize=True):
    # Get all image file paths in the folder, sorted by frame number
    images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))], 
                    key=lambda x: int(x.split('frame_')[1].split('.')[0]))

    # Open images, resize, and reduce color
    frames = []
    for img_path in images:
        img = Image.open(img_path)
        
        # Resize each frame based on the resize_factor
        img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)), Image.Resampling.LANCZOS)
        
        # Optionally reduce colors (to 128 colors)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=128) if optimize else img.convert("RGB")
        
        frames.append(img)

    # Save as compressed GIF
    frames[0].save(
        gif_name,
        save_all=True,
        append_images=frames[1:], 
        duration=duration,
        loop=loop,
        optimize=optimize,
        disposal=2  # Clear each frame after display to reduce flicker
    )
    print(f"Compressed GIF created and saved as {gif_name}")



def visualize_simulation(simulation, lead_bird_radius):
    # Visualization setup
    fig = plt.figure(figsize=(12, 12), facecolor='lightblue')
    ax = fig.add_subplot(111, projection='3d')

    # Ensure the output directory exists
    output_dir = './plot'
    os.makedirs(output_dir, exist_ok=True)

    # Customize appearance
    ax.grid(False)  # Remove grid for a cleaner look
    ax.set_facecolor("white")  # Set background color for better contrast

    # Remove panes and background of 3D plot
    ax.xaxis.pane.set_edgecolor('w')  # Remove the edges of the panes
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.xaxis.pane.set_alpha(0.0)  # Make panes fully transparent
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    # Customizing axis limits
    ax.set_xlim([-2.2 * lead_bird_radius, 2.2 * lead_bird_radius])
    ax.set_ylim([-2.2 * lead_bird_radius, 2.2 * lead_bird_radius])
    ax.set_zlim([0, 3 * lead_bird_radius])

    # Setting labels and title
    ax.set_xlabel("X Axis", fontsize=12, labelpad=20)
    ax.set_ylabel("Y Axis", fontsize=12, labelpad=20)
    ax.set_zlabel("Z Axis", fontsize=12, labelpad=20)
    plt.title("3D Bird Flock Simulation", fontsize=16, color='navy')

    for i, positions in tqdm(enumerate(simulation), ncols=120, total=len(simulation)):
        ax.clear()  # Clear the plot for each frame to avoid overlapping
        
        # Set axis limits and title dynamically per frame
        ax.set_xlim([-1.5 * lead_bird_radius, 1.5 * lead_bird_radius])
        ax.set_ylim([-1.5 * lead_bird_radius, 1.5 * lead_bird_radius])
        ax.set_zlim([0, 3 * lead_bird_radius])
        ax.set_title(f"3D Bird Flock Simulation - Frame {i+1}", fontsize=16, color='darkblue', pad=10)

        # Plot the lead bird with a unique color and marker
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   color='yellow', s=400, alpha=1, edgecolor='darkblue', marker='*', label='Lead Bird')
        
        # Plot the other birds with a lower opacity and shading for a "cloud" effect
        ax.scatter(positions[1:, 0], positions[1:, 1], positions[1:, 2], 
                   color='lightblue', s=20, alpha=0.3, marker='^', edgecolor='darkblue', label='Other Birds')

        # Lighting effect for 3D feel
        ax.view_init(elev=30, azim=135)  # Slight rotation per frame for dynamic effect

        # Add legend
        ax.legend(loc='upper right', fontsize=10)

        # Save each frame
        plt.draw()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/frame_{i:04d}.png', facecolor=fig.get_facecolor())

if __name__ == "__main__":

    # lead_bird_radius = 200.0
    # simulation = pk.load(open('simulation.pk', 'rb'))
    # simulation = simulation[:500]

    # # Visualization setup
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # visualize_simulation(simulation, lead_bird_radius)

    # Usage example
    create_compressed_gif("./plot", gif_name="bird_simulation.gif", duration=100, loop=1, resize_factor=0.3)

    # create_mp4_from_images("./plot", output_video="bird_simulation.mp4", resize_factor=0.7)
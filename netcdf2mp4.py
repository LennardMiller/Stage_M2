import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import sys
sys.path.append('/home/massoale/Stage_M2/Analyse/qgutils-master/')
sys.path.append('/home/massoale/Bureau/Stage_M2/stage_diags/diagnostics_pkg/')
import io_utils as io
import netCDF4 as nc
from matplotlib.colors import Normalize




#Number of simulation
n=519

#choose between 'local' or 'dahu'
where='dahu'


#Reading the netcdf file
if where=='local':
    if n<10:
        simu_name='outdir_000'+str(n)
    elif n<100 and n>=10:
        simu_name='outdir_00'+str(n)
    Path='/home/massoale/Simu_Test/qgw-main/src/'+simu_name+'/'

elif where=='dahu':
    simu_name='dahu_'+str(n)
    Path='/home/massoale/Simu_Test/simu_dahu/simu_dahu'+str(n)+'/outdir_0001/'

elif where=='dahu_job':
    simu_name='dahu_'+str(n)
    Path='simu_dahu'+str(n)+'/outdir_0001/'
else:
    print('Error: where not recognized')
    sys.exit()
print('la simulation chargée est: ' + simu_name )
print("depuis: "+where)

filenames=['/vars.nc']



dataset=nc.Dataset(Path+filenames[0])




t=dataset.variables['time'][:]
x=dataset.variables['x'][:]
y=dataset.variables['y'][:]
#var=dataset.variables['psi'][:,0,:,:]
var=dataset.variables['q'][:,0,:,:]


# Function to create a frame and save it as an image
def create_frame(data, frame_num):
    plt.figure(figsize=(20, 20))
    norm = Normalize(vmin=np.min(data), vmax=np.max(data)/3)
    plt.imshow(data, cmap='ocean', norm=norm)
    
    plt.axis('off')
    filename = f'frame_{frame_num:03d}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filename

# Loop through each frame in your array and create images
frame_files = []
for i in range(var.shape[0]):
    frame_data = var[i, :, :]
    frame_file = create_frame(frame_data, i)
    frame_files.append(frame_file)

# Create a video from the frames
clip = ImageSequenceClip(frame_files, fps=5)  # Adjust FPS as needed
clip.write_videofile(
    'Video/video'+str(simu_name)+'.mp4', 
    codec='libx264', 
    bitrate='5000k',  # Set a higher bitrate
    audio=False,
    threads=4,
    preset='slow'
)

# Clean up frame images if you don't need them anymore
import os
for frame_file in frame_files:
    os.remove(frame_file)
# script to visualize optic flow from driving dataset: tinyurl.com/3ufzcdaa
import io
import os
import re
import csv 
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)
    
def img_from_fig(fig, dpi=180):
    # returns an image as numpy array from figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
def flow_img(file="test.pfm", show=False):
    flow_dims = (150, 150)
    
    rgb_file = file.replace("optical_flow", "frames_finalpass_webp")
    rgb_file = rgb_file.replace("into_future/", "")
    rgb_file = rgb_file.replace("OpticalFlowIntoFuture_", "")
    rgb_file = rgb_file.replace("_L.pfm", ".webp")
    bgr = cv2.imread(rgb_file)
    rgb = bgr[:,:,::-1]
    h, w = rgb.shape[:2]
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    rgb = rgb[:,new_l:new_r,:]
    rgb = cv2.resize(rgb, dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    
    # Creating plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    axes[0].set_title("RGB", y=1.025)
    axes[0].imshow(rgb)
    
    flow, _ = readPFM(file)
    # optical flow is 2D, the z-dim is 0s anyway :)
    flow = flow[:,:,:2]
    
    h, w = flow.shape[:2]
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    flow = flow[:,new_l:new_r,:]
  
    u = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    
    a = np.sqrt((u ** 2) * (v ** 2))
    a = np.log(a + 1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("L2 Norm Optical Flow", y=1.025)
    axes[1].imshow(a)
    
    h, w = flow.shape[:2]
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    flow = flow[:,new_l:new_r,:]
    
    # every 5th element (well, in 1D) so arrow density is not absurd
    u = u[::5, ::5]
    v = v[::5, ::5]
    
    og_u = np.copy(u)
    og_v = np.copy(v)
    u = np.abs(u)
    v = np.abs(v)
    # see extra resolution far away where flow is small
    u = np.log(u + 1) 
    v = np.log(v + 1) 
    # reapply the neg's, which indicate direction of flow
    u[og_u < 0] = u[og_u < 0] * -1
    v[og_v < 0] = v[og_v < 0] * -1
    # arrows were flipped across both axes
    u *= -1
    v *= -1
    
    x = np.arange(0, u.shape[0], 1)
    y = np.arange(0, u.shape[1], 1)
    X, Y = np.meshgrid(x, y)

    # Defining color
    color = np.sqrt(u**2 + v**2).flatten()

    axes[2].axis("off")
    axes[2].set_title("Optical Flow as Vector Field")
    axes[2].quiver(X, Y, u, v, color)
    # trying to make top-left pt 0,0
    axes[2].invert_yaxis()

    if show:
        plt.show()
    
    return img_from_fig(fig)
    
def make_flow_mp4(load_dir="./driving", fps=10, v_name="test.mp4"):
    frames = []
    PFM_dir = os.path.join(load_dir, "optical_flow/15mm_focallength/scene_forwards/slow/into_future/left")
    PFMs = os.listdir(PFM_dir)
    for of_f in sorted(PFMs):
        if of_f.endswith(".pfm"):
            np_img = flow_img(os.path.join(PFM_dir, of_f))
            # trim white borders along left and right side
            np_img = np_img[10:-10,175:-175,:]
            frames.append(np_img)
            # useful for debugging
            # if len(frames) >= 10:
                # break    

    os.makedirs("./tmp", exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame, 'RGB')
        img.save(f"./tmp/rgb_{i:04}.png")
    
    # tried cv2.videoWriter first but would just not work on my ubunut machine :(
    os.system(f"ffmpeg -r {fps} -i ./tmp/rgb_%04d.png -vcodec libx264 -crf 26 -pix_fmt yuv420p -y {v_name}")
    
def dir_response(x, y, θ_pref):
    σ_theta = 3.0
    # matching here: tinyurl.com/jk7kahzd
    angle_x_y = math.atan2(y, x)
    result = np.exp(σ_theta * (math.cos(angle_x_y − θ_pref) − 1)) 
    assert result >= 0 and result <= 1, "speed_response() result out of range!"
    return result

def speed_response(x, y, ρ_pref):
    σ = 1.16
    s0 = 0.33
    # convert from pixels/frame to deg/frame
    deg_per_px = 52.7 / 150
    x *= deg_per_px
    y *= deg_per_px
    # convert from deg/frame to deg/sec
    FPS = 10
    x *= FPS
    y *= FPS
    speed_x_y = np.sqrt(x**2 + y**2)
    result = np.exp(−np.log(speed_x_y + s0 / ρ_pref + s0) ** 2 / 2*σ**2) 
    assert result >= 0 and result <= 1, "speed_response() result out of range!"
    return result
    
def make_flow_csv(load_dir="./driving"):
    flow_dims = (150, 150)
    # units: degrees
    θ_prefs = [0, 45, 90, 135, 180, 225, 270, 315]
    # @TODO is this even the correct xtick values??? Look here:
    # https://github.com/bainro/MSTd_AE_EA/blob/main/matlab_scripts/GenerateInputStim/generateInputStim.m#L9
    # units: degrees / sec
    ρ_prefs = [0.5, 4.375, 8.25, 12.125, 16]
    
    # left & right cameras in both forward & backwards time directions (~3.2k files)
    left_forward = os.path.join(load_dir, "optical_flow/15mm_focallength/scene_forwards/slow/into_future/left")
    PFMs = os.listdir(left_forward)
    right_forward = os.path.join(load_dir, "optical_flow/15mm_focallength/scene_forwards/slow/into_future/right")
    PFMs += os.listdir(right_forward)
    left_backward = os.path.join(load_dir, "optical_flow/15mm_focallength/scene_backwards/slow/into_future/left")
    PFMs += os.listdir(left_backward)
    right_backward = os.path.join(load_dir, "optical_flow/15mm_focallength/scene_backwards/slow/into_future/right")
    PFMs += os.listdir(right_backward)
    rows = []
    for of_f in sorted(PFMs):
        if of_f.endswith(".pfm"):
            trial = []
            flow, _ = readPFM(file)
            flow = flow[:,:,:2]
            # crop to 1:1 aspect ratio
            h, w = flow.shape[:2]
            new_l = round(w/2 - h/2)
            new_r = round(w/2 + h/2)
            flow = flow[:,new_l:new_r,:]
            # reduce image resolution
            u = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
            v = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
            u = u.flatten()
            v = v.flatten()
            # pass the data thru equation 2 to get R_MT (ie responses of all 150x150x40 MT neurons)
            for θ_pref in θ_prefs:
                for ρ_pref in ρ_prefs:
                    for x, y in zip(u, v):
                        # eq 2: R_MT(x, y; θ_pref, ρ_pref) = d(x, y; θ_pref) * s(x, y; ρ_pref)
                        R_MT = dir_response(x, y, θ_pref) * speed_response(x, y, ρ_pref)
                        trial.append(R_MT)
            rows.append(trial)
    
    # will then save into csv wh/ each line is all MT neurons for a "trial"
    with open("./driving-8dir-5speed.csv", 'w') as csv_f: 
        csv_w = csv.writer(csv_f) 
        # csv_w.writerow(fields)  
        csv_w.writerows(rows)
    
if __name__ == "__main__":
    # make_flow_mp4()
    make_flow_csv()

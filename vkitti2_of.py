# script for OF from virtual kitti 2: https://tinyurl.com/57dm7a8j
import io
import os
import re
import csv 
import cv2
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from driving_of import img_from_fig, speed_response, dir_response


def read_OF_png(file):
    # Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array
    # read png to bgr in 16 bit unsigned short

    bgr = cv2.imread(file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0 # or another value (e.g., np.nan)
    return out_flow

def flow_img(file="test.png", f_min=None, f_max=None, show=False):
    flow_dims = (150, 150)
    
    rgb_file = file.replace("forwardFlow", "rgb")
    rgb_file = rgb_file.replace("flow", "rgb")
    rgb_file = rgb_file.replace(".png", ".jpg")
    bgr = cv2.imread(rgb_file)
    rgb = bgr[:,:,::-1]
    h, w = rgb.shape[:2]
    # crop in the sides to achieve a 1:1 aspect ratio
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    rgb = rgb[:,new_l:new_r,:]
    rgb = cv2.resize(rgb, dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    
    # Creating plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    axes[0].set_title("RGB", y=1.025)
    axes[0].imshow(rgb)
    
    flow = read_OF_png(file)
    # normalize the flow so that colors are consistent
    # flow = (flow + f_min) / (f_max + f_min)
    h, w = flow.shape[:2]
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    flow = flow[:,new_l:new_r,:]
  
    u = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    
    a = np.sqrt((u ** 2) + (v ** 2))
    a = np.log(a + 1)
    if f_max > -1 * f_min:
        vmax = np.sqrt((f_max ** 2) + (f_max ** 2)) 
    else:
        vmax = np.sqrt((f_min ** 2) + (f_min ** 2))
    vmax = np.log(vmax + 1)
    
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("L2 Norm Optical Flow", y=1.025)
    axes[1].imshow(a, vmin=0, vmax=vmax)
    
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
    # arrows were flipped across one axis ?
    # u *= -1
    v *= -1
    
    x = np.arange(0, u.shape[0], 1)
    y = np.arange(0, u.shape[1], 1)
    X, Y = np.meshgrid(x, y)

    # Defining color
    # color = np.sqrt(u**2 + v**2).flatten()

    axes[2].axis("off")
    axes[2].set_title("Optical Flow as Vector Field")
    axes[2].quiver(X, Y, u, v, vmin=0, vmax=vmax) # color)
    # trying to make top-left pt 0,0
    axes[2].invert_yaxis()

    if show:
        plt.show()
    
    return img_from_fig(fig)
    
def make_flow_mp4(load_dir="./vkitti2", fps=10, v_name="test.mp4"):
    frames = [] 
    OF_dir = os.path.join(load_dir, "Scene01/sunset/frames/forwardFlow/Camera_0")
    OFs = os.listdir(OF_dir)
    # dumb if ever long term use, but good shortcut
    flow_min = 1e6
    flow_max = -1e6
    for of_f in sorted(OFs):
        flow = read_OF_png(os.path.join(OF_dir, of_f))
        if flow.min() < flow_min:
            flow_min = flow.min()
        if flow.max() > flow_max:
            flow_max = flow.max()
    for of_f in sorted(OFs):
        if of_f.endswith(".png"):
            f = os.path.join(OF_dir, of_f)
            np_img = flow_img(f, flow_min, flow_max)
            # trim white borders along left and right side
            # np_img = np_img[10:-10,175:-175,:]
            frames.append(np_img)
            # useful for debugging
            if len(frames) >= 30:
                break    

    os.makedirs("./tmp/vkitti2", exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame, 'RGB')
        img.save(f"./tmp/vkitti2/rgb_{i:04}.png")
    
    # tried cv2.videoWriter first but would just not work on my ubunut machine :(
    os.system(f"ffmpeg -r {fps} -i ./tmp/vkitti2/rgb_%04d.png -vcodec libx264 -crf 26 -pix_fmt yuv420p -y {v_name}")

def make_flow_csv(load_dir="./driving"):
    # ensures deterministic (thus repeatable) shuffling
    random.seed(42)
    flow_dims = (15, 15)
    # units: degrees
    θ_prefs = [0, 45, 90, 135, 180, 225, 270, 315]
    # units: degrees / sec
    ρ_prefs = [0.5, 4.375, 8.25, 12.125, 16]
    flow_dims = list(flow_dims)
    n_trial_eles = flow_dims[0] * flow_dims[1] * len(θ_prefs) * len(ρ_prefs)
    flow_dims = tuple(flow_dims)
    
    rows = []
    PFMs = []
    # left & right cameras in both forward & backwards time directions (~3.2k files)
    PFM_dirs = ["optical_flow/15mm_focallength/scene_forwards/slow/into_future/left",
                "optical_flow/15mm_focallength/scene_forwards/slow/into_future/right",
                "optical_flow/15mm_focallength/scene_backwards/slow/into_future/left",
                "optical_flow/15mm_focallength/scene_backwards/slow/into_future/right"]
    for dir in PFM_dirs:
        full_path = os.path.join(load_dir, dir)
        for pfm_file in os.listdir(full_path):
            PFMs.append(os.path.join(full_path, pfm_file))
    
    random.shuffle(PFMs)
    for i, of_file in enumerate(PFMs):
        '''
        @TODO: remove as this is a temporary test snippet.
        Putting all 3.2k flow files into a single csv is intractable.
        So going to do just the first 200 to test Kexin's pre-existing csv pipeline,
        before overhauling the C++ to load .npy files instead of a single csv file.
        '''
        # if i > 100:
            # break
        # not sure if necessary, but for my own sanity
        if of_file.endswith(".pfm"):
            trial = []
            flow = read_OF_png(of_file)
            flow = flow[:,:,:2]
            # crop to 1:1 aspect ratio
            h, w = flow.shape[:2]
            new_l = round(w/2 - h/2)
            new_r = round(w/2 + h/2)
            flow = flow[:,new_l:new_r,:]
            # reduce image resolution
            u = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
            v = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
            x = u.flatten()
            y = v.flatten()
            # pass the data thru equation 2 to get R_MT (ie responses of all 15x15x40 MT neurons)
            for θ_pref in θ_prefs:
                for ρ_pref in ρ_prefs:
                    # eq 2: R_MT(x, y; θ_pref, ρ_pref) = d(x, y; θ_pref) * s(x, y; ρ_pref)
                    # @TODO these values are incorrect!
                    R_MT = dir_response(x, y, θ_pref) * speed_response(x, y, ρ_pref, FOV=52.7, orig_h=540, FPS=8)
                    trial += R_MT.tolist()
            assert len(trial) == n_trial_eles, f"{len(trial)} != {n_trial_eles}"
            rows.append(trial)
    
    # will then save into csv wh/ each line is all MT neurons for a "trial"
    with open("./driving-8dir-5speed.csv", 'w') as csv_f: 
        csv_w = csv.writer(csv_f) 
        # csv_w.writerow(fields)  
        csv_w.writerows(rows)
    
if __name__ == "__main__":
    make_flow_mp4()
    # make_flow_csv()

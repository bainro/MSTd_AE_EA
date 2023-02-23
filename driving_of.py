# script for optic flow from driving dataset: tinyurl.com/3ufzcdaa 
# download for final_pass & OF tar files: tinyurl.com/228emu8s
import io
import os
import re
import csv 
import cv2
import math
import random
import imagehash
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import genfromtxt


def csv_stats(filenames):
    """
    def readCSV(filename):
        with open(filename, "r") as csv_f:
            csv_r = csv.reader(csv_f)
            for row in csv_r:
                # remove the "[" and "]"
                del row[0]; del row[-1]
                row = ", ".join(row)
                '''
                print(row)
                print(type(row))
                print(len(row))
                '''
                row_as_list = row.split(", ")
                '''
                print(row_as_list)
                print(type(row_as_list))
                print(len(row_as_list))
                '''
                row_as_list = [float(i) for i in row_as_list]
                yield row_as_list
                
    for filename in filenames:
        print(f"csv_stats({filename})")    
        min_v = 1000
        max_v = -1000
        running_total = 0       
        num_rows = 0
        for row in readCSV(filename):
            num_rows = num_rows + 1
            running_total = running_total + sum(row)
            if min(row) < min_v:
                min_v = min(row)
            elif max(row) > max_v:
                max_v = max(row)
        
        print(f"shape: {num_rows} x {len(row)}")
        print(f"avg value: {running_total / num_rows}")
        print(min_v, max_v)
        print("====================")
    """
        
    for filename in filenames:
        print(f"csv_stats({filename})")    
        of_data = genfromtxt(filename, delimiter=',')
        print(f"shape: {of_data.shape}")
        print(f"avg value: {np.mean(of_data)}")
        print(f"max value: {np.max(of_data)}")
        print(f"min value: {np.min(of_data)}")
        print(f"value stdev: {np.std(of_data)}")
        print("====================")

    '''
    if len(filenames) == 2:
        # show histogram of values
        fig, ax = plt.subplots()
        for filename in filenames:
            of_data = genfromtxt(filename, delimiter=',')
            of_sorted = np.sort(of_data.flatten())
            plt.plot(of_sorted)
        plt.show()
    '''
        
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
    flow_dims = (15, 15) # (150, 150)
    
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
    
    a = np.sqrt((u ** 2) + (v ** 2))
    a = np.log(a + 1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("L2 Norm Optical Flow", y=1.025)
    # not a perfectly general solution, but just for visualizations
    vmax = 5.3
    if vmax < a.max():
        # warning instead of error as some outliers are tolerable to keep scale
        # print(f"Warning. vmax < a.max(): {a.max()}")
        pass
    axes[1].imshow(a, vmin=0, vmax=vmax)
    
    h, w = flow.shape[:2]
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    flow = flow[:,new_l:new_r,:]
    
    # Used this for dims of 150x150, but turned off for 15x15
    # every 5th element (well, in 1D) so arrow density is not absurd
    #u = u[::5, ::5]
    #v = v[::5, ::5]
    
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
    # arrows were flipped across one axis?
    # u *= -1
    v *= -1
    
    x = np.arange(0, u.shape[0], 1)
    y = np.arange(0, u.shape[1], 1)
    X, Y = np.meshgrid(x, y)

    # Defining color as only black
    color = np.zeros(u.shape).flatten()
    # will need to change for different resolutions
    scale = 60
    
    axes[2].axis("off")
    axes[2].set_title("Optical Flow as Vector Field")
    axes[2].quiver(X, Y, u, v, color, scale=scale)
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
            ### Was originally for 150x150, but tolerating it now for debugging
            # trim white borders along left and right side
            # np_img = np_img[10:-10,175:-175,:]
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
    angle_x_y = np.arctan2(y, x)
    result = np.exp(σ_theta * (np.cos(angle_x_y - θ_pref) - 1)) 
    # assert result >= 0 and result <= 1, "dir_response() result out of range!"
    return result

def speed_response(x, y, ρ_pref):
    σ = 1.16
    s0 = 0.33
    speed_x_y = np.sqrt(x**2 + y**2)
    
    # Nover 2005 paper seems to have meant natural log for the modeling eq's but log10 for axis?
    result = np.exp(-np.log((speed_x_y + s0) / (ρ_pref + s0)) ** 2 / 2*σ**2) 
    # assert result >= 0 and result <= 1, "speed_response() result out of range!"
    return result
    
def make_flow_csv(load_dir="./driving"):
    # ensures deterministic (thus repeatable) shuffling
    random.seed(42)
    # height x width
    flow_dims = (225, 225)
    flow_dims = list(flow_dims)
    # units: degrees
    θ_prefs = [0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]
    ρ_prefs = [0.0087, 0.0208, 0.0494, 0.1174, 0.2793]
    
    win_len = 15
    n_trial_eles = win_len * win_len * len(θ_prefs) * len(ρ_prefs)
    flow_dims = tuple(flow_dims)
    
    PFMs = []
    # left & right cameras in both forward & backwards time directions (~3.2k files)
    PFM_dirs = ["optical_flow/15mm_focallength/scene_forwards/slow/into_future/right",]
    ''' # @TODO remove! for debug only! rows matrix was too big, giving OOM err
    "optical_flow/15mm_focallength/scene_forwards/slow/into_future/left","
    "optical_flow/15mm_focallength/scene_forwards/slow/into_future/right",
    "optical_flow/15mm_focallength/scene_backwards/slow/into_future/left",
    "optical_flow/15mm_focallength/scene_backwards/slow/into_future/right"]
    '''
    for dir in PFM_dirs:
        full_path = os.path.join(load_dir, dir)
        # @TODO remove! for debug only!
        num_flow_files = len(os.listdir(full_path))
        for q, pfm_file in enumerate(os.listdir(full_path)):
            if q > num_flow_files // 800: # 16:
                break
            PFMs.append(os.path.join(full_path, pfm_file))
    
    # conv windowed input overlap (width - stride)
    overlap = 2
    stride = win_len - overlap
    n_p_o = math.floor(flow_dims[0] / (win_len - overlap)) 
    print(f"ratio of new windowed inputs per old, whole input: {n_p_o ** 2}")
    n_conv_windows = len(PFMs) * n_p_o ** 2
    rows = np.zeros((n_conv_windows, n_trial_eles))
    row_i = 0
    # random.shuffle(PFMs)
    for i, of_file in enumerate(PFMs):
        
        # not sure if necessary, but for my own sanity
        if of_file.endswith(".pfm"):
            flow, _ = readPFM(of_file)
            flow = flow[:,:,:2]
            # crop to 1:1 aspect ratio
            h, w = flow.shape[:2]
            new_l = round(w/2 - h/2)
            new_r = round(w/2 + h/2)
            flow = flow[:,new_l:new_r,:]
            # reduce image resolution
            u = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
            v = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
            # had to flip for movie, might need to for csv too.
            # not certain yet whether x needs to be flipped too.
            v *= -1
            
            x = np.flip(u, 0)
            y = np.flip(v, 0)

            # double check that format is HxW elsewhere in the code if this fails!
            assert flow_dims[0] == flow_dims[1]
            # subsampling input using j and k
            for j in range(n_p_o):
                for k in range(n_p_o):
                    trial = []
                    prev_hash = None
                    # pass the data thru equation 2 to get R_MT (ie responses of all 15x15x40 MT neurons)
                    for ρ_pref in ρ_prefs:
                        for θ_pref in θ_prefs:
                            _j = stride * j
                            _k = stride * k 
                            _x = x[_j:(_j + win_len), _k:(_k + win_len)]
                            _y = y[_j:(_j + win_len), _k:(_k + win_len)]
                            if θ_pref == 0:
                                hash = imagehash.average_hash(Image.fromarray( np.uint8(np.dstack((_x,_y)) * 255) ))
                                if prev_hash != None:
                                    print(hash - prev_hash)
                                prev_hash = hash
                            if k == n_p_o - 1:
                                exit()
                            _x = _x.flatten()
                            _y = _y.flatten()
                            # eq 2: R_MT(x, y; θ_pref, ρ_pref) = d(x, y; θ_pref) * s(x, y; ρ_pref)
                            R_MT = dir_response(_x, _y, θ_pref) * speed_response(_x, _y, ρ_pref)	
                            trial += R_MT.tolist()	
                    assert len(trial) == n_trial_eles, f"{len(trial)} != {n_trial_eles}"	
                       
                    rows[row_i, :] = np.array(trial)
                    row_i = row_i + 1
                    
    # will then save into csv wh/ each line is all MT neurons for a "trial"
    with open("/media/rbain/aa31c0ce-f5cd-4b96-8d9d-58b2507995e7/driving-8dir-5speed.csv", 'w') as csv_f: 
        csv_w = csv.writer(csv_f) 
        rows = rows.T
        print("rows.shape: " + str(rows.shape))
        csv_w.writerows(rows)
    
if __name__ == "__main__":
    # make_flow_mp4(os.environ['HOME'] + "/driving_data")
    make_flow_csv('/home/rbain/driving_data')
    csv_stats(["driving-8dir-5speed.csv", "V-8dir-5speed.csv"])

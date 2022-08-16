# script to visualize optic flow from driving dataset: tinyurl.com/3ufzcdaa
import io
import os
import re
import cv2
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
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
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
    
    x = np.arange(0, u.shape[0], 1)
    y = np.arange(0, u.shape[1], 1)
    X, Y = np.meshgrid(x, y)

    # Defining color
    color = np.sqrt(u**2 + v**2).flatten()

    axes[2].axis("off")
    axes[2].quiver(X, Y, u, v, color)
    # trying to make top-left pt 0,0
    axes[2].invert_yaxis()

    if show:
        plt.show()
    
    return img_from_fig(fig)
    
def make_flow_mp4(load_dir="./driving", fps=10, v_name="test.avi"):
    
    frames = []
    PFM_dir = os.path.join(load_dir, "optical_flow/15mm_focallength/scene_forwards/fast/into_future/left")
    PFMs = os.listdir(PFM_dir)
    for of_f in sorted(PFMs):
        if of_f.endswith(".pfm"):
            np_img = flow_img(os.path.join(PFM_dir, of_f))
            frames.append(np_img)
            if len(frames) >= 10:
                break    

    os.mkdir("./tmp")
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame, 'RGB')
        img.save(f"./tmp/rgb_{i:04}.png")
    
    
if __name__ == "__main__":
    make_flow_mp4()

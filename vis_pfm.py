# script to visualize optic flow from driving dataset: tinyurl.com/3ufzcdaa

import re
import cv2
import numpy as np
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
    
def vis_img_flow(file="test.pfm"):

    flow, _ = readPFM(file)
    # optical flow is 2D, the z-dim is 0s anyway :)
    flow = flow[:,:,:2]
    
    h, w = flow.shape[:2]
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    flow = flow[:,new_l:new_r,:]
   
    flow_dims = (150, 150)
    u = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    
    a = np.sqrt((u ** 2) * (v ** 2))
    a = np.log(a + 1)
    _ = plt.imshow(a)

    plt.show()
    
def vis_quiver_flow(file="test.pfm"):
    
    flow, _ = readPFM(file)
    # optical flow is 2D, the z-dim is 0s anyway :)
    flow = flow[:,:,:2]
    
    h, w = flow.shape[:2]
    new_l = round(w/2 - h/2)
    new_r = round(w/2 + h/2)
    flow = flow[:,new_l:new_r,:]
    
    flow_dims = (150, 150)
    u = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    
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

    # Creating plot
    fig, ax = plt.subplots()
    ax.quiver(X, Y, u, v, color)
    # trying to make top-left pt 0,0
    plt.gca().invert_yaxis()

    # show plot
    plt.show()
    
def make_flow_mp4(load_dir, fps, v_name):
    """
    load_dir: where the .PFMs and .WEBPs are saved
    fps: Desired frames per second
    v_name: what to name the new video
    """
    
    img_files = os.path.list_dir(load_dir)
    frames = []
    for fin img_files:
        if f.endswith(".png"):
            tmp_f = os.path.join([img_dir, f])
            frames.append(cv2.imread(tmp_f))

    w, h = frames[0].size
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release() 
    
    
if __name__ == "__main__":
    make_flow_mp4()

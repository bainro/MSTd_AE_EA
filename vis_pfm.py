# script to visualize optic flow from driving dataset: tinyurl.com/3ufzcdaa

import re
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
    print(flow.shape);exit()
    
    import cv2
    flow_dims = (150, 150)
    flow[:,:,0] = cv2.resize(flow[:,:,0], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    flow[:,:,1] = cv2.resize(flow[:,:,1], dsize=flow_dims, interpolation=cv2.INTER_CUBIC)
    
    x = np.arange(0, flow.shape[0], 1)
    y = np.arange(0, flow.shape[1], 1)
    X, Y = np.meshgrid(x, y)
    
    u = np.copy(flow[:,:,0])
    v = np.copy(flow[:,:,1])
    
    a = np.sqrt((u ** 2) * (v ** 2))
    a = np.log(a + 1)
    _ = plt.imshow(a)

    plt.show()
    
def vis_quiver_flow(file="test.pfm"):
    
    # @TODO remove hardcoded img dims!
    x = np.arange(0, 960, 1)
    y = np.arange(0, 540, 1)

    X, Y = np.meshgrid(x, y)

    flow, _ = readPFM(file)
    # optical flow is 2D, the z-dim is 0s anyway :)
    flow = flow[:,:,:2]        
    u = np.copy(flow[:,:,0])
    v = np.copy(flow[:,:,1])
    u = np.abs(u)
    v = np.abs(v)
    # see extra resolution far away where flow is small
    u = np.log(u + 1) 
    v = np.log(v + 1) 

    # Defining color
    color = 1 # np.sqrt(((dx-n)/2)*2 + ((dy-n)/2)*2)

    # Creating plot
    fig, ax = plt.subplots()
    ax.quiver(X, Y, u, v, color)

    # show plot
    plt.show()
    
if __name__ == "__main__":
    vis_quiver_flow()

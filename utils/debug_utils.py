import torchvision
from torchviz import make_dot
# Used for debugging
def save_tensor_img(img, name='rendering'):
    torchvision.utils.save_image(img, name+".png")

def save_cal_graph(var,name='cal_graph'):
    dot = make_dot(var)
    dot.format = 'png'
    dot.render(filename=name, directory='./', cleanup=True) 
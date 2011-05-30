import numpy as N
import sys
from pylearn2.gui import patch_viewer
from pylearn2.config import yaml_parse

assert len(sys.argv) == 2
path = sys.argv[1]

if path.endswith('.pkl'):
    from framework.utils import serial
    dataset = serial.load(path)
elif path.endswith('.yaml'):
    dataset =yaml_parse.load_path(path)
else:
    dataset = yaml_parse.load(path)

rows = 20
cols = 20

examples = dataset.get_batch_topo(rows*cols)

print ('examples range',examples.min(),examples.max(), examples.dtype)

examples /= N.abs(examples).max()

if len(examples.shape) != 4:
    print 'sorry, view_examples.py only supports image examples for now.'
    print 'this dataset has '+str(len(examples)-2)+' topological dimensions'
    quit(-1)
#

if examples.shape[3] == 1:
    is_color = False
elif examples.shape[3] == 3:
    is_color = True
else:
    print 'got unknown image format with '+str(examples.shape[3])+' channels'
    print 'supported formats are 1 channel greyscale or three channel RGB'
    quit(-1)
#

print examples.shape[1:3]

pv = patch_viewer.PatchViewer( (rows, cols), examples.shape[1:3], is_color = is_color)

for i in xrange(rows*cols):
    pv.add_patch(examples[i,:,:,:], rescale = False)
#

pv.show()

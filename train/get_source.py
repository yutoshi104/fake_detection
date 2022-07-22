import inspect
import numpy
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

with open('./src/src_image.py','w') as f:
    f.write(inspect.getsource(image))

with open('./src/src_image_generator.py','w') as f:
    f.write(inspect.getsource(ImageDataGenerator))

with open('./src/src_sequence.py','w') as f:
    f.write(inspect.getsource(Sequence))

#from dlgo.data.processor import GoDataProcessor
from data.parallel_processor import GoDataProcessor
import numpy as np

processor = GoDataProcessor()
generator = processor.load_go_data('train', 100,use_generator=True)

for features, labels in generator.generate():
    print(features.shape)
    print(labels.shape)

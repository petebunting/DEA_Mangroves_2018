import glob
import rsgislib.imageutils

h5Files = glob.glob('./ExtractedCoverPVStats/*.h5')

rsgislib.imageutils.mergeExtractedHDF5Data(h5Files, './AllExtractedCoverPVStats_V5.0.h5')


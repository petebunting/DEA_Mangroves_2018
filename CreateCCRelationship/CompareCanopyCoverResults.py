import pandas
import h5py
import numpy
import matplotlib.pyplot as plt
import glob
import os.path

h5Files = glob.glob('./ExtractedCoverPVStats/*.h5')

aggData = dict()
for h5File in h5Files:
    print(h5File)
    baseName = os.path.splitext(os.path.basename(h5File))[0].replace('_edit', '').replace('_Stats', '').replace('_', ' ')
    baseName = baseName.replace('WAlligator', 'West Alligator')
    
    fH5 = h5py.File(h5File)
    data = fH5['DATA/DATA']
    
    pvData = data[...,0]
    pvData = pvData.astype(int)
    ccData = data[...,1]
    
    dataValsFrame = pandas.DataFrame([pvData,ccData], index=['GVp10','pcc']).T

    aggCCDF = dataValsFrame.groupby('GVp10').agg({'pcc': ['mean','std']})
    
    xVals = aggCCDF.index.values
    yVals = aggCCDF['pcc']['mean'].values
    yErrVals = aggCCDF['pcc']['std'].values
    aggData[baseName] = [xVals, yVals, yErrVals]
    fH5.close()


ax = plt.subplot(111)
markers = ["^", "+", "x", "o", "D", 'd', "*", "h","8", "_", "|", "1", "2", "3", "4"]
i = 0
for site in aggData:
    print(site)
    ax.scatter(aggData[site][0], aggData[site][1], color='#808080', label=site, marker=markers[i])
    #ax.errorbar(aggData[site][0],aggData[site][1],yerr=aggData[site][2], linestyle="None")
    i = i + 1
ax.legend()
ax.set_title('GVpc10 vs PCC')
ax.set_xlabel('GVpc10 (%)')
ax.set_ylabel('Planimetric Canopy Cover (%)')

plt.savefig('SiteComparisonCanopyCover.pdf')
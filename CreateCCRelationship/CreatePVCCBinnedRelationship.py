import pandas
import h5py
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker
import random

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model

######## READ DATA ############

h5File = './AllExtractedCoverPVStats_V5.0.h5'

fH5 = h5py.File(h5File)
data = fH5['DATA/DATA']

pvData = data[...,0]
pvData = pvData.astype(int)
ccData = data[...,1]
fH5.close()

##############################


######## DATA INFO ############
print(pvData.shape)
print(ccData.shape)

print('pv min: ', numpy.min(pvData))
print('pv max: ', numpy.max(pvData))

print('cc min: ', numpy.min(ccData))
print('cc max: ', numpy.max(ccData))
##############################


######## CREATE DATA FRAME AND SPLIT INTO TRAIN AND VALIDATE ############
dataValsFrame = pandas.DataFrame([pvData,ccData], index=['GVp10','pcc']).T

ids = dataValsFrame.index.values.astype(int)
random.shuffle(ids)

n = int(len(ids)/2)
ids1, ids2 = ids[:n], ids[n:]

dfTrain = dataValsFrame[dataValsFrame.index.isin(ids1)]
print('Train: ', dfTrain.shape)
dfValid = dataValsFrame[dataValsFrame.index.isin(ids2)]
print('Validate: ', dfValid.shape)
##############################


######## BIN DATA TO GEN RELATIONSHIP ############
aggCCDF = dfTrain.groupby('GVp10').agg({'pcc': ['mean','std']})

xVals = aggCCDF.index.values
yVals = aggCCDF['pcc']['mean'].values
errVals = aggCCDF['pcc']['std'].values
errVals[numpy.isnan(errVals)] = 0
print('Mean Bin Std Dev: ', numpy.mean(errVals))
##############################
"""
######## PLOT RESULT ############
ax = plt.subplot(111)
ax.scatter(xVals, yVals, color='#808080', zorder=10)
ax.errorbar(xVals,yVals,yerr=errVals, color='#E0E0E0', linestyle="None", zorder=1)

ax.set_title('GVpc10 vs Binned PCC')
ax.set_xlabel('GVpc10 (%)')
ax.set_ylabel('Planimetric Canopy Cover (%)')
ax.set_xlim(0,100)
ax.set_ylim(0,100)

plt.savefig('CanopyCoverGVpc10_v_BinPCC_errbars.pdf')
##############################
"""


######## FIT MODEL ############
model = sklearn.pipeline.Pipeline([('poly', sklearn.preprocessing.PolynomialFeatures(degree=3)), ('linear', sklearn.linear_model.LinearRegression(fit_intercept=False))])

model.fit(xVals.reshape(-1, 1), yVals)

rsq = model.score(xVals.reshape(-1, 1), yVals)
print('r sq = ', rsq)

pvStep = numpy.arange(0, 100, 1)
ccSteps = model.predict(pvStep.reshape(-1, 1))
##############################


######## CREATE THRESHOLD LUT ############
print('GVp10,\tPCC')
for i in range(len(pvStep)):
    print(str(pvStep[i]) + '\t' + str(round(ccSteps[i], 2)))

# Threshold look up table
print('PV : ', pvStep[15])
print('CC : ', ccSteps[15])

print('PV : ', pvStep[40])
print('CC : ', ccSteps[40])

print('PV : ', pvStep[62])
print('CC : ', ccSteps[62])
##############################


######## TEST MODEL WITH VALID DF ############
ccCalcd = model.predict(dfValid['GVp10'].values.reshape(-1, 1))
rmsePxls = ((dfValid.pcc - ccCalcd) ** 2).mean() ** .5
print('RMSE to Pixels: ', rmsePxls)


aggCCDFValid = dfValid.groupby('GVp10').agg({'pcc': ['mean','std']})
xValsValid = aggCCDFValid.index.values
yValsValid = aggCCDFValid['pcc']['mean'].values
yPredVals = model.predict(xValsValid.reshape(-1, 1))
rmseBins = ((yValsValid - yPredVals) ** 2).mean() ** .5
print('RMSE to Groups: ', rmseBins)

##############################

"""
######## PLOT MODEL RESULT ############
ax = plt.subplot(111)
ax.plot(pvStep, ccSteps, color='black', zorder=20)
ax.scatter(xVals, yVals, color='#808080', zorder=10)
ax.errorbar(xVals,yVals,yerr=errVals, color='#E0E0E0', linestyle="None", zorder=1)

ax.set_title('GVpc10 vs Binned PCC')
ax.set_xlabel('GVpc10 (%)')
ax.set_ylabel('Planimetric Canopy Cover (%)')
ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.text(2, 95, r'$r^2$='+str(round(rsq,4)), fontsize=10)
ax.text(2, 90, r'RMSE='+str(round(rmseBins,1))+' %', fontsize=10)
ax.text(2, 85, r'Pixel RMSE='+str(round(rmsePxls,1))+' %', fontsize=10)
plt.savefig('CanopyCoverGVpc10_v_BinPCC_relationship_errbars.pdf')

##############################
"""

"""
######## PLOT MODEL RESULT THRESHOLDS SHADED ############
ax = plt.subplot(111)
ax.plot(pvStep, ccSteps, color='black', zorder=50)
ax.scatter(xVals, yVals, color='#808080', zorder=20)
ax.errorbar(xVals,yVals,yerr=errVals, color='#E0E0E0', linestyle="None", zorder=10)
ax.fill([0, 0, 40, 40, 15, 15], [ccSteps[15], ccSteps[40], ccSteps[40], 0, 0, ccSteps[15]], facecolor='#FFFF66', alpha=0.2, zorder=100)
ax.fill([0, 0, 62, 62, 40, 40], [ccSteps[40], ccSteps[62], ccSteps[62], 0, 0, ccSteps[40]], facecolor='#80FF00', alpha=0.2, zorder=100)
ax.fill([0, 0, 100, 100, 62, 62], [ccSteps[62], 100, 100, 0, 0, ccSteps[62]], facecolor='#006633', alpha=0.2, zorder=100)

ax.text(25, 1, r'Woodland', fontsize=10, zorder=200)
ax.text(44, 1, r'Open Forest', fontsize=10, zorder=200)
ax.text(80, 1, r'Closed Forest', fontsize=10, zorder=200)

ax.set_title('GVpc10 vs Binned PCC')
ax.set_xlabel('GVpc10 (%)')
ax.set_ylabel('Planimetric Canopy Cover (%)')
ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax.text(2, 95, r'$r^2$='+str(round(rsq,4)), fontsize=10, zorder=200)
ax.text(2, 90, r'RMSE='+str(round(rmseBins,1))+' %', fontsize=10, zorder=200)
ax.text(2, 85, r'Pixel RMSE='+str(round(rmsePxls,1))+' %', fontsize=10, zorder=200)
plt.savefig('CanopyCoverGVpc10_v_BinPCC_relationship_errbars_shadeForestThres.pdf')

##############################
"""

"""
######## PREDICTED VS KNOWN ############

plt.figure(1, figsize=(10,8))

nullfmt = matplotlib.ticker.NullFormatter() 

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(dfValid.pcc, ccCalcd, marker=".", s=2, zorder=1)
axScatter.plot([0,100], [0,100], zorder=10, color='black')

# now determine nice limits by hand:
binwidth = 5
xymax = max(numpy.max(numpy.abs(dfValid.pcc)), numpy.max(numpy.abs(ccCalcd)))
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((0, 102))
axScatter.set_ylim((0, 102))

bins = numpy.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(dfValid.pcc, bins=bins)
axHisty.hist(ccCalcd, bins=bins, orientation='horizontal')

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

axHistx.set_title('PCC Predicted Verses Validation')
axScatter.set_xlabel('LiDAR Planimetric Canopy Cover (%)')
axScatter.set_ylabel('Landsat Planimetric Canopy Cover (%)')

plt.savefig('LandsatPCC_relationship_PredVsKnown.png')
##############################
"""



######## PREDICTED VS KNOWN ############
binRange = numpy.arange(0, 100, 1)
#print(binRange)

knownPCC = numpy.array(dfValid['pcc'])
predPCC = numpy.array(ccCalcd)

#print(knownPCC)
#print(predPCC)

knownPCCGrpd = numpy.zeros_like(binRange, dtype=float)
predPCCGrpd = numpy.zeros_like(binRange, dtype=float)

knownPCCGrpdStd = numpy.zeros_like(binRange, dtype=float)
predPCCGrpdStd = numpy.zeros_like(binRange, dtype=float)

for bin in binRange:
    #print(str(bin) + ' to ' + str(bin+1))
    tmpKnowPCC = knownPCC[(predPCC>=bin) & (predPCC<(bin+1))]
    tmpPredPCC = predPCC[(knownPCC>=bin) & (knownPCC<(bin+1))]
    if tmpKnowPCC.shape[0] > 0:
        knownPCCGrpd[bin] = numpy.mean(tmpKnowPCC)
        knownPCCGrpdStd[bin] = numpy.std(tmpKnowPCC)
    if tmpPredPCC.shape[0] > 0:
        predPCCGrpd[bin] = numpy.mean(tmpPredPCC)
        predPCCGrpdStd[bin] = numpy.std(tmpPredPCC)

binRangeEdit = binRange[predPCCGrpd>0]
knownPCCGrpdStd = knownPCCGrpdStd[predPCCGrpd>0]
predPCCGrpdStd = predPCCGrpdStd[predPCCGrpd>0]
knownPCCGrpd = knownPCCGrpd[predPCCGrpd>0]
predPCCGrpd = predPCCGrpd[predPCCGrpd>0]

binRangeEdit = binRangeEdit[knownPCCGrpd>0]
knownPCCGrpdStd = knownPCCGrpdStd[knownPCCGrpd>0]
predPCCGrpdStd = predPCCGrpdStd[knownPCCGrpd>0]
predPCCGrpd = predPCCGrpd[knownPCCGrpd>0]
knownPCCGrpd = knownPCCGrpd[knownPCCGrpd>0]


print(knownPCCGrpdStd)
#print(predPCCGrpdStd)
from sklearn.metrics import r2_score
coefficient_of_dermination = r2_score(binRangeEdit, knownPCCGrpd)
rmse = ((binRangeEdit - knownPCCGrpd) ** 2).mean() ** .5

ax1 = plt.subplot(111)
ax1.scatter(binRangeEdit, knownPCCGrpd, zorder=100)
#ax.errorbar(knownPCCGrpd, predPCCGrpd,xerr=knownPCCGrpdStd, yerr=predPCCGrpdStd, color='#E0E0E0', linestyle="None", zorder=10)
ax1.plot([0,100], [0,100], zorder=110, color='black')
ax1.set_xlim(0,100)
ax1.set_ylim(0,100)
ax1.set_title('PCC Predicted Verses Training')
ax1.set_xlabel('LiDAR Binned Planimetric Canopy Cover (%)')
ax1.set_ylabel('Landsat Mean Planimetric Canopy Cover (%)')

ax1.text(2, 94, r'$r^2$='+str(round(coefficient_of_dermination,4)), fontsize=10, zorder=200)
ax1.text(2, 89, r'RMSE='+str(round(rmse,1))+' %', fontsize=10, zorder=200)

plt.savefig('LandsatPCC_relationship_PredVsKnown_binned.pdf')
##############################








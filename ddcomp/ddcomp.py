import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
import numpy as np
import math
from matplotlib.ticker import AutoMinorLocator
import sys

def sigmoid(x):
    return 1/(1 + math.exp(-x))
    
def hyperbolic(x,k):
    return 1/(1+k*x)
    
def heaviside(x,k):
    if x<k:
        return 0
    else:
        return 1
        
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    
def dataComp(csvName,futureVal,outputDir='./dataOutput/'):
    columnNames = ['Participant ID','Delay Period','Delay Value',
        'Present Value','Choice']
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    df = pd.read_csv(csvName,header=None,names=columnNames)
    ps = df['Participant ID'].unique()
    numParts = len(ps)

    cond = df['Delay Period'].unique()
    maxCond = max(cond)
    minCond = min(cond)
    nc = len(cond)
    condDic = {n:cond[n] for n in range(0,nc)}

    # Create files if not existing.  Truncate if existing.
    f = open(outputDir + 'Summary_Hyperbolic_Data.txt','w+')
    f.close()
    f = open(outputDir + 'Summary_Hyperbolic_Data_NoFlags.txt','w+')
    f.close()
    f = open(outputDir + 'Summary_Hyperbolic_Data_AllFlagged.txt','w+')
    f.close()
    f = open(outputDir + 'Summary_Hyperbolic_Data.csv','w+')
    f.close()
    
    f = open(outputDir + 'Summary_Hyperbolic_Data.csv','a')
    f.write('Participant, k-Value\n')
    f.close()

    totIter = len(ps)*len(cond)
    updateP = 0
    printProgressBar(updateP, totIter, prefix = 'Progress:', suffix = 'Complete', length = 50)
    ps_hyperbolic_k = []
    for p in ps:
        inflectionVals = []
        flag = 0
        flaggedPeriod = set()
        for c in cond:
            updateP += 1
            printProgressBar(updateP, totIter, prefix = 'Progress:', suffix = 'Complete', length = 50)
            tempDf = df[((df['Delay Period'] == c)&(df['Participant ID'] == p))]
            tempDf = tempDf.reset_index(drop = True)

            xval = tempDf['Present Value'].astype(int).values.tolist()
            minX = min(xval)
            maxX = max(xval)
            minXSpread = range(min(xval)-5,min(xval))
            maxXSpread = range(max(xval)+1,max(xval)+5)
            xval.extend(minXSpread)
            xval.extend(maxXSpread)
            yval = tempDf['Choice'].astype(int).values.tolist()
            yval.extend([0,0,0,0,0])
            yval.extend([1,1,1,1])
            
            if not os.path.exists(outputDir + str(p)):
                os.makedirs(outputDir + str(p))
            
            X = np.array(xval)
            X = X[:,np.newaxis]
            clf = LogisticRegression(C=1e5)
            clf.fit(X,yval)
            
            Xtest = np.linspace(minX,maxX,1000)
            modelY = sigmoid(Xtest * clf.coef_ + clf.intercept_).ravel()
            theInflection = int(math.ceil(-1*clf.intercept_/clf.coef_))
            
            fig,ax = plt.subplots()
            ax.set_axisbelow(True)

            xVert = np.linspace(theInflection,theInflection,num=100)
            yVert = np.linspace(0,1,num=100)
            xHorz = np.linspace(0,theInflection,num=100)
            
            fileSave = outputDir + str(p) + '/' + 'p' + str(p) + '_' + 'c' + str(c) + '.png'
            plotOut = DD_Plotter(maxX)
            plotOut.fill(xHorz)
            plotOut.plotV(xVert,yVert)
            plotOut.plot(Xtest,modelY)
            plotOut.scatter(tempDf['Present Value'].astype(int).values.tolist(),tempDf['Choice'].astype(int).values.tolist())
            plotOut.axes('Present Value',r'Preference (1$\equiv$Immediate)')
            plotOut.save(fileSave)
            
            if theInflection < 0:
                inflectionVals.append(0)
            elif theInflection > maxX:
                inflectionVals.append(maxX)
            else:
                inflectionVals.append(theInflection)
            
            sensitivityObs = 0.0
            sensitivityPre = 0.0
            specificityObs = 0.0
            specificityPre = 0.0
            for i in range(0,9):
                xval.pop()
                yval.pop()
            for i,x in enumerate(xval):
                if yval[i] == 1:
                    sensitivityObs += 1
                    if heaviside(x,theInflection) == 1:
                        sensitivityPre += 1
                elif yval[i] == 0:
                    specificityObs += 1
                    if heaviside(x,theInflection) == 0:
                        specificityPre += 1
            
            if sensitivityObs == 0:
                sensitivity = 0
            else:
                sensitivity = sensitivityPre/sensitivityObs
                
            if specificityObs == 0:
                specificity = 0
            else:
                specificity = specificityPre/specificityObs
            
            if sensitivity <=0.5 and sensitivity != 0.0:
                flag = 1
                flaggedPeriod.add(c)
            elif specificity <=0.5 and specificity != 0.0:
                flag = 1
                flaggedPeriod.add(c)
            
            f = open(outputDir + str(p) + '/inflections.txt','a')
            f.write('Delay Period: ')
            f.write(str(c))
            f.write(', Inflection Point: ')
            f.write(str(theInflection))
            f.write(', Sensitivity: %01.2f'%sensitivity)
            f.write(', Specificity: %01.2f'%specificity)
            f.write('\n')
            f.close()
        
        xHyper = [condDic[key] for key in condDic]
        maxInflection = max(inflectionVals)
        if maxInflection == 0:
            maxInflection = 1
        minInflection = 0
        yHyper = [float(y)/futureVal for y in inflectionVals]
        
        f = open(outputDir + str(p) + '/inflections.txt','a')
        if np.mean(yHyper) == 0 or np.mean(yHyper) == 1:
            f.write('Hyperbolic k: invalid, Rsq: invalid')
            f.close()
            
            f = open(outputDir + 'Summary_Hyperbolic_Data.txt','a')
            f2 = open(outputDir + 'Summary_Hyperbolic_Data_AllFlagged.txt','a')
            if flag == 1:
                f.write('Participant: %d, Hyperbolic k: invalid, Rsq: invalid, --Check Data (Condition(s): '%p)
                f2.write('Participant: %d, Hyperbolic k: invalid, Rsq: invalid, --Check Data (Condition(s): '%p)
                for i,item in enumerate(flaggedPeriod):
                    if i<len(flaggedPeriod)-1:
                        f.write(str(item))
                        f.write(', ')
                        f2.write(str(item))
                        f2.write(', ')
                    else:
                        f.write(str(item))
                        f.write(')\n')
                        f2.write(str(item))
                        f2.write(')\n')
            else:
                f.write('Participant: %d, Hyperbolic k: invalid, Rsq: invalid\n'%p)
                f2.write('Participant: %d, Hyperbolic k: invalid, Rsq: invalid\n'%p)
            f.close()
            f2.close()
        else:
            popt, pcov = curve_fit(hyperbolic,xHyper,yHyper)
            ps_hyperbolic_k.append(popt[0])
            
            residuals = yHyper - hyperbolic(xHyper,popt)
            ssRes = np.sum(residuals**2)
            ssTot = np.sum((yHyper - np.mean(yHyper))**2)
            rSq = 1 - (ssRes/(ssTot+1e-10))
        
            f.write('Hyperbolic k: %04.3f, Rsq: %04.2f'%(popt[0],rSq))
            f.close()
            
            f = open(outputDir + 'Summary_Hyperbolic_Data.txt','a')
            
            if flag == 1:
                f2 = open(outputDir + 'Summary_Hyperbolic_Data_AllFlagged.txt','a')
                f.write('Participant: %d, Hyperbolic k: %04.3f, Rsq: %04.2f, --Check Data (Condition(s): '%(p,popt[0],rSq))
                f2.write('Participant: %d, Hyperbolic k: %04.3f, Rsq: %04.2f, --Check Data (Condition(s): '%(p,popt[0],rSq))
                for i,item in enumerate(sorted(flaggedPeriod)):
                    if i<len(flaggedPeriod)-1:
                        f.write(str(item))
                        f.write(', ')
                        f2.write(str(item))
                        f2.write(', ')
                    else:
                        f.write(str(item))
                        f.write(')\n')
                        f2.write(str(item))
                        f2.write(')\n')
            else:
                f.write('Participant: %d, Hyperbolic k: %04.3f, Rsq: %04.2f\n'%(p,popt[0],rSq))
                f2 = open(outputDir + 'Summary_Hyperbolic_Data_NoFlags.txt','a')
                f2.write('Participant: %d, Hyperbolic k: %04.3f, Rsq: %04.2f\n'%(p,popt[0],rSq))
                f2.close()
            f.close()
            
            f = open(outputDir + 'Summary_Hyperbolic_Data.csv','a')
            f.write(str(p))
            f.write(',')
            f.write('%04.3f'%popt[0])
            f.write('\n')
            f.close()
            
            xHyperTest = np.linspace(0,max(xHyper),1000)
            yHyperTest = [hyperbolic(x,popt[0]) for x in xHyperTest]
            
            fileSave = outputDir + str(p) + '/' + 'p' + str(p) + '_hyperbolic.png'
            plotOut = DD_Plotter(maxCond,hyperbolic=True)
            plotOut.axes('Delay Period','Preference Inflection Value (normalized)')
            plotOut.plot(xHyperTest,yHyperTest)
            plotOut.scatter(xHyper,yHyper)
            plotOut.save(fileSave)

    mean_k = np.mean(ps_hyperbolic_k)
    xHyperTest = np.linspace(0,max(xHyper),1000)
    yHyperTest = [hyperbolic(x,mean_k) for x in xHyperTest]
   
    plotOut = DD_Plotter(maxCond,hyperbolic=True)
    plotOut.axes('Delay Period','Perference Inflection Value (normalized)','Group Average Discounting Function')
    plotOut.plot(xHyperTest,yHyperTest)
    plotOut.save(outputDir + 'Group_Average_Hyperbolic.png')
    
class DD_Plotter():
    def __init__(self,maxCond,hyperbolic=False):
        self.fig, self.ax = plt.subplots(figsize=(8,5.5))
        self.ax.set_axisbelow(True)
        self.ax.set_ylim([-0.05,1.05])
        self.ax.set_xlim([0,maxCond])
        self.ax.set_yticks([0,0.5,1])
        if not hyperbolic:
            self.xlabs = range(0,maxCond,2)
            self.ax.set_xticks(self.xlabs)

        self.minor_locator = AutoMinorLocator(2)
        self.ax.xaxis.set_minor_locator(self.minor_locator)

        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.ax.yaxis.set_ticks_position('left')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.grid(which='major',linestyle='-.',color='#C9C9C9')
        self.ax.grid(which='minor',linestyle='--',color='#EAEAEA')

    def axes(self,xlab,ylab,title=''):
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
    
    def plot(self,x,y):
        self.ax.plot(x,y,color='#F72C8B',linewidth=3,zorder=2)
        
    def plotV(self,x,y):
        self.ax.plot(x,y,color='#686EE2',zorder=2,linewidth=2,linestyle='--')
        
    def scatter(self,x,y):
        self.ax.scatter(x,y,facecolor='#252525',edgecolor='face',s=2**6,clip_on=False,zorder=3)
    
    def fill(self,x):
        self.ax.fill_between(x,0,1,color='#686EE2',alpha=0.1,zorder=1)
        
    def save(self,pathVal):
        plt.savefig(pathVal,bbox_inches='tight')
        plt.close('all')
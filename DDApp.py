import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import FixedTicker
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit
import os
import math

class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.initUI()
        
    def initUI(self):
        titleFont = QFont('SansSerif',20)
        titleFont.setBold(True)
        
        vbox = QVBoxLayout()
        vbox.setSpacing(10)
        
        self.btn = QPushButton('Select Data File')
        self.btn.clicked.connect(self.fileOpen)
        
        self.closeBtn = QPushButton('Close')
        self.closeBtn.setStyleSheet("background-color: #E26161")
        self.closeBtn.clicked.connect(QCoreApplication.instance().quit)
        
        self.prgLabel = QLabel('Progress')
        self.prgLabelDone = QLabel('')
        
        self.prg = QProgressBar()
        self.prg.setMinimum(0)
        self.prg.setMaximum(100)
        self.prg.setValue(0)
        
        self.titleLabel = QLabel('Delay Discounting App')
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setFont(titleFont)
        
        vbox.addWidget(self.titleLabel)
        vbox.addSpacing(30)
        vbox.addWidget(self.btn)
        vbox.addSpacing(10)
        vbox.addWidget(self.prgLabel)
        vbox.addWidget(self.prg)
        vbox.addSpacing(10)
        vbox.addWidget(self.prgLabelDone)
        vbox.addWidget(self.closeBtn)
        
        self.setLayout(vbox)
        
        self.resize(500,100)
        self.setWindowTitle('Delay Discounting')
        self.center()
        self.show()
    
    def fileOpen(self):
        name = QFileDialog.getOpenFileName(self,'Select Data File')
        try:
            self.btn.setEnabled(False)
            self.closeBtn.setEnabled(False)
            self.dataOutput(str(name))
            self.closeBtn.setEnabled(True)
        except:
            self.btn.setEnabled(True)
            self.closeBtn.setEnabled(True)
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def updatePRG(self,val):
        self.prg.setValue(val)
        
    def dataOutput(self,csvName):
        columnNames = ['Participant ID','Delay Period','Delay Value','Present Value','Choice']

        df = pd.read_csv(csvName,header=None,names=columnNames)
        ps = df['Participant ID'].unique()
        numParts = len(ps)
        psDic = {n:ps[n] for n in range(0,numParts)}

        cond = df['Delay Period'].unique()
        nc = len(cond)
        condDic = {n:cond[n] for n in range(0,nc)}

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
            
        def hyperbolic(x,k):
            return 1/(1+k*x)
            
        def heaviside(x,k):
            if x<k:
                return 0
            else:
                return 1

        f = open('hyperbolicData.txt','w')
        f.close()
        loopCount = 0.0
        for p in ps:
            inflectionVals = []
            for c in cond:
                QApplication.processEvents() 
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
                
                outputFilename = str(p) + '/' + str(p) + '_' + str(c) + '.html'
                output_file(outputFilename,title='DD Decision',mode='inline')
                
                if not os.path.exists('./' + str(p)):
                    os.makedirs('./' + str(p))
                
                X = np.array(xval)
                X = X[:,np.newaxis]
                clf = LogisticRegression(C=1e5)
                clf.fit(X,yval)
                
                Xtest = np.linspace(minX,maxX,1000)
                modelY = sigmoid(Xtest * clf.coef_ + clf.intercept_).ravel()
                
                figTitle = 'p' + str(p) + '_' + 'c' + str(c) + ' DD Inflection'
                fig = figure(title=figTitle)
                fig.xaxis.axis_label = 'Present Value'
                fig.yaxis.axis_label = 'Choice'
                fig.yaxis[0].ticker = FixedTicker(ticks=[0,0.5,1])
                fig.ygrid.grid_line_dash = [6, 4]
                fig.xgrid.grid_line_dash = [6, 4]
                
                fig.line(Xtest,modelY,line_width=3,line_color = '#F72C8B')
                fig.scatter(tempDf['Present Value'].astype(int).values.tolist(),tempDf['Choice'].astype(int).values.tolist(),fill_color='#000000',radius=0.25,line_color = None)
                save(fig)
                
                theInflection = int(math.ceil(-1*clf.intercept_/clf.coef_))
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
                
                loopCount += 1
                self.prg.setValue((loopCount/(numParts*len(cond)))*100)
                
                f = open('./' + str(p) + '/inflections.txt','a')
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
            yHyper = [float(y)/maxInflection for y in inflectionVals]
            
            f = open('./' + str(p) + '/inflections.txt','a')
            if np.mean(yHyper) == 0 or np.mean(yHyper) == 1:
                f.write('Hyperbolic k: invalid, Rsq: invalid')
                f.close()
                
                f = open('hyperbolicData.txt','a')
                f.write('Participant: %d, Hyperbolic k: invalid, Rsq: invalid\n'%p)
                f.close()
            else:
                popt, pcov = curve_fit(hyperbolic,xHyper,yHyper)
                
                residuals = yHyper - hyperbolic(xHyper,popt)
                ssRes = np.sum(residuals**2)
                ssTot = np.sum((yHyper - np.mean(yHyper))**2)
                rSq = 1 - (ssRes/(ssTot+1e-10))
            
                f.write('Hyperbolic k: %04.3f, Rsq: %04.2f'%(popt[0],rSq))
                f.close()
                
                f = open('hyperbolicData.txt','a')
                f.write('Participant: %d, Hyperbolic k: %04.3f, Rsq: %04.2f\n'%(p,popt[0],rSq))
                f.close()
                
                f = open('hyperbolicDataCSV.csv','a')
                f.write(str(p))
                f.write(',')
                f.write('%04.3f'%popt[0])
                f.write('\n')
                f.close()
                
                xHyperTest = np.linspace(0,max(xHyper),1000)
                yHypertest = [hyperbolic(x,popt[0]) for x in xHyperTest]

                outputFilename = str(p) + '/' + str(p) + '_' + 'hyperbolic.html'
                output_file(outputFilename,title='DD Hyperbolic',mode='inline')
                
                fig = figure()
                fig.line(xHyperTest,yHypertest,line_width=3,line_color = '#F72C8B')
                fig.scatter(xHyper,yHyper,fill_color='#000000',radius=2.5,line_color=None)
                save(fig)
            
        self.prgLabelDone.setText('Complete!')
        
def main():
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
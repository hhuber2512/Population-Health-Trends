import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sci
import sklearn
from sklearn.linear_model import LinearRegression

'#create Obesity array by interpolation. Intepolate by step one to generate sizable dataset(45 vs. 23 points'
xorig = [1971,1977,1991,1999,2001,2003,2005,2007,2009,2011,2013,2015]
yorig = [.14,.15,.23,.31,.31,.33,.34,.34,.36,.35,.37,.4]

def interpolate(xorig,yorig):
    f = sci.interpolate.interp1d(xorig,yorig,'linear')
    xrange = np.arange(1971,2016,1)
    y = f(xrange)
    return [xrange,y]

(yearobese,percentobese) = interpolate(xorig,yorig)

'#get Sugar DF'
def pulldf(filepath):
    DF = pd.read_excel(filepath)
    return DF

SugarDF = pulldf('/Users/hollyhuber/Documents/1. Projects/Obesity Trends/First Round-NHANES/Sugar Trends'
                 '/Sweetener Production.xlsx')

'#Generate Lagged Array of Sugar Production - List Version'
 def generatelagmatrix(yearobese,YearSug,Sugar,lag):
    matrix = []
    yearobeselist = yearobese.tolist()
    for year in yearobeselist:
        yearlylaglist = []
        yearlyvaluelist = []
        for size in lag:
            yearlag = year-size
            yearlylaglist.append(yearlag)
        for year in yearlylaglist:
            index = YearSug[YearSug == year].index[0]
            value = Sugar[index]
            yearlyvaluelist.append(value)
        matrix.append(yearlyvaluelist)
    matrixdf = pd.DataFrame(matrix)
    return matrixdf

'#apply lag matrix generating function'
lag = range(0,40,1)
HFCS = generatelagmatrix(yearobese,SugarDF['Year'],SugarDF['Corn Sweeteners(lb per cap)'],lag)
AllSugar = generatelagmatrix(yearobese,SugarDF['Year'],SugarDF['Total Sweeteners(lb per cap)'],lag)

def regression1(X,Y):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.22222,
                                                                                shuffle=False)
    model = LinearRegression().fit(X_train, Y_train)
    prediction_train = model.predict(X_train)
    residuals_train = np.subtract(Y_train,prediction_train)
    mean_square_error_train = np.sum(np.square(residuals_train))/len(residuals_train)
    prediction_test = model.predict(X_test)
    residuals_test = np.subtract(Y_test, prediction_test)
    mean_square_error_test = np.sum(np.square(residuals_test)) / len(residuals_test)
    coefficients = model.coef_
    intercept = model.intercept_
    prediction = np.append(prediction_train,prediction_test)
    return mean_square_error_train,mean_square_error_test,coefficients,intercept,prediction


'#Single Term Linear Regression'
'#X is sugar df, y is percent obese'
def singleoptimal(X,Y,lag):
    trainerr_oneterm = []
    testerr_oneterm = []
    index_oneterm = []
    coef_oneterm = []
    intercept_oneterm = []
    prediction_oneterm = []
    for index in lag:
        X_array = X[[index]]
        trainerr,testerr,coeff,intercept,pred = regression1(X_array,Y)
        trainerr_oneterm.append(trainerr)
        testerr_oneterm.append(testerr)
        index_oneterm.append(index)
        coef_oneterm.append(coeff)
        intercept_oneterm.append(intercept)
        prediction_oneterm.append(pred)
    mintest = min(testerr_oneterm)
    index_mintest = testerr_oneterm.index(mintest)
    mintrain = min(trainerr_oneterm)
    index_mintrain = trainerr_oneterm.index(mintrain)
    print('Model Optimized on Train Error, train error:', trainerr_oneterm[index_mintrain])
    print('Model Optimized on Train Error, test error:', testerr_oneterm[index_mintrain])
    print('Model Optimized on Train Error, indices:', index_oneterm[index_mintrain])
    print('Model Optimized on Train Error,coefficients:', coef_oneterm[index_mintrain])
    print('Model Optimized on Train Error, intercept:', intercept_oneterm[index_mintrain])
    print('Model Optimized on Test Error, train error:', trainerr_oneterm[index_mintest])
    print('Model Optimized on Test Error, test error:', testerr_oneterm[index_mintest])
    print('Model Optimized on Test Error, indices:', index_oneterm[index_mintest])
    print('Model Optimized on Test Error,coefficients:', coef_oneterm[index_mintest])
    print('Model Optimized on Test Error, intercept:', intercept_oneterm[index_mintest])
    return trainerr_oneterm,testerr_oneterm,index_oneterm,coef_oneterm,intercept_oneterm,prediction_oneterm,index_mintest,\
           index_mintrain

trainerr_oneterm,testerr_oneterm,index_oneterm,coef_oneterm,intercept_oneterm,prediction_oneterm,index_mintest,index_mintrain\
    = singleoptimal(HFCS,percentobese,lag)

'#Visualizing Single Term Regression'
plt.figure()
plt.xlabel('Index of Lagged Array')
plt.ylabel('Mean Square Error')
plt.title('Test/Train Mean Square Error Single Term Regression HFCS Predictor')
plt.plot(index_oneterm,testerr_oneterm,'o',label='Test Error, min @ (t-13)')
plt.plot(index_oneterm,trainerr_oneterm,'o',label = 'Train Error, min @ (t-10)')
plt.legend()

plt.figure()
plt.xlabel('Index of Lagged Array')
plt.ylabel('Mean Square Error')
plt.title('Test Mean Square Error Single Term Regression')
plt.plot(index_oneterm,testerr_oneterm,'o',label='Min @ index=13')
plt.legend()

plt.figure()
plt.xlabel('Index of Lagged Array')
plt.ylabel('Mean Square Error')
plt.title('Train Mean Square Error Single Term Regression')
plt.plot(index_oneterm,trainerr_oneterm,'o')

plt.figure()
plt.xlabel('Time, Years')
plt.ylabel('Percent Obese')
plt.title('Testing Set Optimal Prediction, HFCS, One Term')
plt.plot(yearobese,percentobese,'o',label='Empirical Obesity')
plt.plot(yearobese,prediction_oneterm[13],'o',label='Predicted')
plt.axvline(x=2005,label='test train split')
plt.legend()

'#Two Term Linear Regression'
def doubleoptimal(X,Y,lag):
    trainerr_twoterm = []
    testerr_twoterm = []
    index1_twoterm = []
    index2_twoterm = []
    coef_twoterm = []
    intercept_twoterm = []
    prediction_twoterm = []
    for index1 in lag:
        for index2 in lag:
            X_array = X[[index1,index2]]
            trainerr, testerr, coeff, intercept, pred = regression1(X_array, Y)
            trainerr_twoterm.append(trainerr)
            testerr_twoterm.append(testerr)
            index1_twoterm.append(index1)
            index2_twoterm.append(index2)
            coef_twoterm.append(coeff)
            intercept_twoterm.append(intercept)
            prediction_twoterm.append(pred)
    mintest = min(testerr_twoterm)
    index_mintest = testerr_twoterm.index(mintest)
    mintrain = min(trainerr_twoterm)
    index_mintrain = trainerr_twoterm.index(mintrain)
    print('Model Optimized on Train Error, train error:', trainerr_twoterm[index_mintrain])
    print('Model Optimized on Train Error, test error:', testerr_twoterm[index_mintrain])
    print('Model Optimized on Train Error, indices:', index1_twoterm[index_mintrain], index2_twoterm[index_mintrain])
    print('Model Optimized on Train Error,coefficients:', coef_twoterm[index_mintrain])
    print('Model Optimized on Train Error, intercept:', intercept_twoterm[index_mintrain])
    print('Model Optimized on Test Error, train error:', trainerr_twoterm[index_mintest])
    print('Model Optimized on Test Error, test error:', testerr_twoterm[index_mintest])
    print('Model Optimized on Test Error, indices:', index1_twoterm[index_mintest], index2_twoterm[index_mintest])
    print('Model Optimized on Test Error,coefficients:', coef_twoterm[index_mintest])
    print('Model Optimized on Test Error, intercept:', intercept_twoterm[index_mintest])
    return trainerr_twoterm, testerr_twoterm, index1_twoterm, index2_twoterm, coef_twoterm, intercept_twoterm, \
           prediction_twoterm, index_mintest,index_mintrain


trainerr_twoterm, testerr_twoterm, index1_twoterm, index2_twoterm, coef_twoterm, intercept_twoterm, \
prediction_twoterm, index_mintest, index_mintrain = doubleoptimal(AllSugar,percentobese,lag)

plt.figure()
plt.xlabel('Time, Years')
plt.ylabel('Percent Obese')
plt.title('Testing Set Optimal Prediction, All Sugar, Two Term')
plt.plot(yearobese,percentobese,'o',label='Empirical Obesity')
plt.plot(yearobese,prediction_twoterm[220],'o',label='Predicted')
plt.axvline(x=2005,label='test train split')
plt.legend()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('index1')
ax.set_ylabel('index2')
ax.set_zlabel('Mean Square Error')
ax.set_title('Errors of Two Term Linear Regression Optimization,All Sugar')
ax.plot_trisurf(index1_twoterm, index2_twoterm, testerr_twoterm, label='Test Error')
ax.plot_trisurf(index1_twoterm,index2_twoterm,trainerr_twoterm,label='Train Error')

'#Three-Term Linear Regression'
def tripleoptimal(X,Y,lag):
    trainerr_threeterm = []
    testerr_threeterm = []
    index1_threeterm = []
    index2_threeterm = []
    index3_threeterm = []
    coef_threeterm = []
    intercept_threeterm = []
    prediction_threeterm = []
    for index1 in lag:
        for index2 in lag:
            for index3 in lag:
                X_array = X[[index1,index2,index3]]
                trainerr, testerr, coeff, intercept, pred = regression1(X_array, Y)
                trainerr_threeterm.append(trainerr)
                testerr_threeterm.append(testerr)
                index1_threeterm.append(index1)
                index2_threeterm.append(index2)
                index3_threeterm.append(index3)
                coef_threeterm.append(coeff)
                intercept_threeterm.append(intercept)
                prediction_threeterm.append(pred)
    mintest = min(testerr_threeterm)
    index_mintest = testerr_threeterm.index(mintest)
    mintrain = min(trainerr_threeterm)
    index_mintrain = trainerr_threeterm.index(mintrain)
    print('Model Optimized on Train Error, train error:', trainerr_threeterm[index_mintrain])
    print('Model Optimized on Train Error, test error:', testerr_threeterm[index_mintrain])
    print('Model Optimized on Train Error, indices:', index1_threeterm[index_mintrain], index2_threeterm[index_mintrain],
          index3_threeterm[index_mintrain])
    print('Model Optimized on Train Error,coefficients:', coef_threeterm[index_mintrain])
    print('Model Optimized on Train Error, intercept:', intercept_threeterm[index_mintrain])
    print('Model Optimized on Test Error, train error:', trainerr_threeterm[index_mintest])
    print('Model Optimized on Test Error, test error:', testerr_threeterm[index_mintest])
    print('Model Optimized on Test Error, indices:', index1_threeterm[index_mintest], index2_threeterm[index_mintest],
          index3_threeterm[index_mintest])
    print('Model Optimized on Test Error,coefficients:', coef_threeterm[index_mintest])
    print('Model Optimized on Test Error, intercept:', intercept_threeterm[index_mintest])
    return trainerr_threeterm, testerr_threeterm, index1_threeterm, index2_threeterm, index3_threeterm, coef_threeterm, intercept_threeterm, \
           prediction_threeterm, index_mintest,index_mintrain


trainerr_threetermHFCS, testerr_threetermHFCS, index1_threetermHFCS, index2_threetermHFCS, index3_threetermHFCS, coef_threetermHFCS, intercept_threetermHFCS, \
           prediction_threetermHFCS, index_mintestHFCS,index_mintrainHFCS = tripleoptimal(HFCS,percentobese,lag)

'#Four Term Linear Regression'
def quadoptimal(X,Y,lag):
    trainerr_fourterm = []
    testerr_fourterm = []
    index1_fourterm = []
    index2_fourterm = []
    index3_fourterm = []
    index4_fourterm = []
    coef_fourterm = []
    intercept_fourterm = []
    prediction_fourterm = []
    for index1 in lag:
        for index2 in lag:
            for index3 in lag:
                for index4 in lag:
                    X_array = X[[index1,index2,index3,index4]]
                    trainerr, testerr, coeff, intercept, pred = regression1(X_array, Y)
                    trainerr_fourterm.append(trainerr)
                    testerr_fourterm.append(testerr)
                    index1_fourterm.append(index1)
                    index2_fourterm.append(index2)
                    index3_fourterm.append(index3)
                    index4_fourterm.append(index4)
                    coef_fourterm.append(coeff)
                    intercept_fourterm.append(intercept)
                    prediction_fourterm.append(pred)
    mintest = min(testerr_fourterm)
    index_mintest = testerr_fourterm.index(mintest)
    mintrain = min(trainerr_fourterm)
    index_mintrain = trainerr_fourterm.index(mintrain)
    print('Model Optimized on Train Error, train error:', trainerr_fourterm[index_mintrain])
    print('Model Optimized on Train Error, test error:', testerr_fourterm[index_mintrain])
    print('Model Optimized on Train Error, indices:', index1_fourterm[index_mintrain], index2_fourterm[index_mintrain],
          index3_fourterm[index_mintrain],index4_fourterm[index_mintrain])
    print('Model Optimized on Train Error,coefficients:', coef_fourterm[index_mintrain])
    print('Model Optimized on Train Error, intercept:', intercept_fourterm[index_mintrain])
    print('Model Optimized on Test Error, train error:', trainerr_fourterm[index_mintest])
    print('Model Optimized on Test Error, test error:', testerr_fourterm[index_mintest])
    print('Model Optimized on Test Error, indices:', index1_fourterm[index_mintest], index2_fourterm[index_mintest],
          index3_fourterm[index_mintest],index4_fourterm[index_mintest])
    print('Model Optimized on Test Error,coefficients:', coef_fourterm[index_mintest])
    print('Model Optimized on Test Error, intercept:', intercept_fourterm[index_mintest])
    return trainerr_fourterm, testerr_fourterm, index1_fourterm, index2_fourterm, index3_fourterm, index4_fourterm, coef_fourterm, intercept_fourterm, \
           prediction_fourterm, index_mintest,index_mintrain

trainerr_fourtermHFCS, testerr_fourtermHFCS, index1_fourtermHFCS, index2_fourtermHFCS, index3_fourtermHFCS, index4_fourtermHFCS, coef_fourtermHFCS, intercept_fourtermHFCS, \
           prediction_fourtermHFCS, index_mintestHFCS,index_mintrainHFCS = quadoptimal(HFCS,percentobese,lag)

trainerr_fourterm, testerr_fourterm, index1_fourterm, index2_fourterm, index3_fourterm, index4_fourterm, coef_fourterm, intercept_fourterm, \
           prediction_fourterm, index_mintest,index_mintrain = quadoptimal(AllSugar,percentobese,lag)


'# test test test'


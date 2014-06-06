'''
Created on Jan 18, 2013

@author: kahere
'''

def Cubic_spline(x):
    y[y==1] = np.nan
    y_spline = np.append(y[1:],min_freq)
    x_spline = np.append(x[1:],50)
    y_spline = np.append(y_spline,1)
    x_spline = np.append(x_spline,500)
    valid_ind = np.where(~np.isnan(y_spline))
    y_spline = y_spline[valid_ind]
    x_spline = x_spline[valid_ind]
    spline_y = scipy.interpolate.UnivariateSpline(x_spline,y_spline,s=smooth)
    x2 = np.linspace(0,20,(20+1)*5)
    y2 = interp_y(x2)
    y3 = spline_y(x2)
    y2[y2 > 1] = 1
    y3[y3 > 1] = 1
    y3[y3 < 0] = 0
    y3 = np.sort(y3)

if __name__ == '__main__':
    from scipy import linspace
    import statsmodels.api as sm
    
    from pylab import plot, show
    import matplotlib.pyplot as plt
    import numpy as np
    
    from scipy.interpolate import interp1d
    import scipy.interpolate
    
    import random
    import bisect
    
    np.set_printoptions(threshold=np.nan)
    
    # Linear SLR
    annual_rise = 0.00932081826307457
    MSL_2012 = 5.856592026
    
    
    # Departure from mean sea level
    anom = np.loadtxt(r'C:\Python code\Hurricane\src\root\nested\BatteryAnom2.txt')
    n = np.size(anom)    
    x = linspace(0,20,n) 
#    x = np.append(x,500)

    smooth = 0.005
    min_freq = 0.9999
    # Sandy return = 119 yrs when smooth = 0.005, min_freq = 0.9999
    
    # Exceedance probability based on event size only (anom)
    ecdf = sm.distributions.ECDF(anom)
    y = ecdf(x)

    interp_y = interp1d(x,y,kind = 'cubic')
    y[y==1] = np.nan
    y_spline = np.append(y[1:],min_freq)
    x_spline = np.append(x[1:],50)
    y_spline = np.append(y_spline,1)
    x_spline = np.append(x_spline,500)
    valid_ind = np.where(~np.isnan(y_spline))
    y_spline = y_spline[valid_ind]
    x_spline = x_spline[valid_ind]
    spline_y = scipy.interpolate.UnivariateSpline(x_spline,y_spline,s=smooth)
    x2 = np.linspace(0,20,(20+1)*5)
    y2 = interp_y(x2)
    y3 = spline_y(x2)
    y2[y2 > 1] = 1
    y3[y3 > 1] = 1
    y3[y3 < 0] = 0
    y3 = np.sort(y3)
    print ("Anomaly")
    print (1/(1-y2[np.where(x2-anom[0] == np.min(np.abs(x2-anom[0])))[0]-1])[0]/12)
    print (1/(1-y3[np.where(x2-anom[0] == np.min(np.abs(x2-anom[0])))[0]-1])[0]/12)
#    plot(x2,y2)
    plot(x2,y3)
    plt.ylim((-.1,1.1))
    y_anom = y3
    
    show()
    
    years = [1900, 1950, 1960, 2012, 2030, 2050, 2100]
    test_height = 13.3 # maximum measured sea level height (includes MSL)  NYC subways flood at 13.78
    test_rp = 50 # return period in years
    test_rp = test_rp*12 # convert return period from years to months
    for i in years:
        ecdf = sm.distributions.ECDF(anom + (MSL_2012 - ((2012-i) * annual_rise)))
        y = ecdf(x)
#        y[y==1] = np.nan
#        y_spline = np.append(y[1:],min_freq)
#        x_spline = np.append(x[1:],50)
#        y_spline = np.append(y_spline,1)
#        x_spline = np.append(x_spline,500)
#        valid_ind = np.where(~np.isnan(y_spline))
#        y_spline = y_spline[valid_ind]
#        x_spline = x_spline[valid_ind]
#        spline_y = scipy.interpolate.UnivariateSpline(x_spline,y_spline,s=smooth)
#        x2 = np.linspace(0,20,(20+1)*5)
#        y2 = interp_y(x2)
#        y3 = spline_y(x2)
#        y2[y2 > 1] = 1
#        y3[y3 > 1] = 1
#        y2 = y3
        y2=y
        x2=x
        print ("Year", i)
#        print 1/(1-y[np.where(x-(anom[0] + MSL_2012) == np.min(np.abs(x-(anom[0] + MSL_2012))))[0]])[0]
#        print 1/(1-y2[np.argmin(np.abs(x2-(anom[0] + MSL_2012)))])
        print ("Height for", test_rp/12, "year return period:", x2[np.argmin(np.abs(1/(1-y2)-test_rp))]) # Height of water given test_rp
        print ("Return period for", test_height, "ft water height:", 1/(1-y2[np.argmin(np.abs(x2-test_height))])/12) # Changing return interval for given test_height
        plot(x2,y2)
        
    plt.ylim((-.1,1.1))
    show()
    
# http://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability
    def choice(population,weights):
#        assert len(population) == len(weights)
#        cdf_vals=cdf(weights)
        assert len(population) == len(weights)
        cdf_vals = y_anom
        x=random.random()
        idx=bisect.bisect(cdf_vals,x)
        return population[idx]

    population=np.linspace(0,30,(30+1)*100)
    counts = np.zeros((np.size(population),2))
    counts[:,0] = population
    
    ecdf = sm.distributions.ECDF(anom)
    y = ecdf(x)

    y[y==1] = np.nan
    y_spline = np.append(y[1:],min_freq)
    x_spline = np.append(x[1:],50)
    y_spline = np.append(y_spline,1)
    x_spline = np.append(x_spline,500)
    valid_ind = np.where(~np.isnan(y_spline))
    y_spline = y_spline[valid_ind]
    x_spline = x_spline[valid_ind]
    spline_y = scipy.interpolate.UnivariateSpline(x_spline,y_spline,s=smooth)
    x2 = population
    y3 = spline_y(x2)
    y3[y3 > 1] = 1
    y3[y3 < 0] = 0
    y_anom = np.sort(y3)
    
    for i in range(10000):
        counts[np.where(counts[:,0] == choice(population,y_anom))[0],1]+=1
#    print(counts)
    
    test_cdf = np.zeros((np.size(population),1))
    for i in range(len(counts)):
        test_cdf[i] = np.sum(counts[:i,1])
    plot(population,test_cdf/10000.)
    plot(population,y_anom,'r-')
    
    show()
    
    MSL = np.loadtxt(r'C:\Documents and Settings\KAHERE\My Documents\Python code\Hurricane\src\root\nested\Battery2100MSL.txt')
    MSL = MSL[:np.size(MSL,0)/2,:]
    monte = 10
    test_rp_height = np.zeros((monte,1))
    test_height_rp = np.zeros((monte,1))
    for k in range(monte):
        MSL_proj = np.zeros((np.size(MSL,0),2))
        MSL_proj[:,0] = MSL[:,0]
        MSL_proj[:,1] = map(lambda i: MSL[i,1] + choice(population,y_anom),range(np.size(MSL_proj,0)))
                   
        ecdf = sm.distributions.ECDF(MSL_proj[:,1])
        y = ecdf(population)
        
#        y[y==1] = np.nan
#        y_spline = np.append(y[1:],min_freq)
#        x_spline = np.append(population[1:],50)
#        y_spline = np.append(y_spline,1)
#        x_spline = np.append(x_spline,500)
#        valid_ind = np.where(~np.isnan(y_spline))
#        y_spline = y_spline[valid_ind]
#        x_spline = x_spline[valid_ind]
#        spline_y = scipy.interpolate.UnivariateSpline(x_spline,y_spline,s=smooth)
#        y3 = spline_y(population)
#        y3[y3 > 1] = 1
#        y3[y3 < 0] = 0
#        y = np.sort(y3)
        
        test_rp_height[k,0] = population[np.argmin(np.abs(1/(1-y)-test_rp))]
        test_height_rp[k,0] = 1/(1-y[np.argmin(np.abs(population-test_height))])/12
    
#    plot(population,y)
#    show()
    plot(MSL_proj[:,0],MSL_proj[:,1])
    plt.hlines(13.78,2013,2112,colors='k',linestyles='dotted')
    plt.hlines(17.33,2013,2112,colors='r',linestyles='dashed')
    show()
    print ("Height for", test_rp/12, "year return period:", np.mean(test_rp_height), "+/-", np.std(test_rp_height))
    print ("Return period for", test_height, "ft water height:", np.mean(test_height_rp[np.where(~np.isinf(test_height_rp))]), "+/-", np.std(test_height_rp[np.where(~np.isinf(test_height_rp))]))

    percentile = [5, 10, 25, 50, 75, 90, 95]
    for i in percentile:
        print (i, "th Percentile")
        print ("Height for", test_rp/12, "year return period:", np.percentile(test_rp_height,i))
        print ("Return period for", test_height, "ft water height:", np.percentile(test_height_rp[np.where(~np.isinf(test_height_rp))],i))
        
    monte = 10000
    test_rp_height_2013 = np.zeros((monte,1))
    test_height_rp_2013 = np.zeros((monte,1))
   
    for m in range(monte):
        height = np.zeros((monte))
        for k in range(monte):
            # SIMULATE 2013 10K TIMES
            height[k] = MSL_2012 + annual_rise + choice(population,y_anom)
            
        
        
        
        
                   
        ecdf = sm.distributions.ECDF(height)
        y = ecdf(population)
        
#        y[y==1] = np.nan
#        y_spline = np.append(y[1:],min_freq)
#        x_spline = np.append(population[1:],50)
#        y_spline = np.append(y_spline,1)
#        x_spline = np.append(x_spline,500)
#        valid_ind = np.where(~np.isnan(y_spline))
#        y_spline = y_spline[valid_ind]
#        x_spline = x_spline[valid_ind]
#        spline_y = scipy.interpolate.UnivariateSpline(x_spline,y_spline,s=smooth)
#        y3 = spline_y(population)
#        y3[y3 > 1] = 1
#        y3[y3 < 0] = 0
#        y = np.sort(y3)
        
        plot(population,y)
        test_rp_height_2013[m] = population[np.argmin(np.abs(1/(1-y)-test_rp))]
        test_height_rp_2013[m] = 1/(1-y[np.argmin(np.abs(population-test_height))])/12    
    
    percentile = [5, 10, 25, 50, 75, 90, 95]
    for i in percentile:
        print (i, "th Percentile")
        print ("Height for", test_rp/12, "year return period:", np.percentile(test_rp_height_2013,i))
        print ("Return period for", test_height, "ft water height (max 90.667):", np.percentile(test_height_rp_2013[np.where(~np.isinf(test_height_rp_2013))],i))
        
    show()
        
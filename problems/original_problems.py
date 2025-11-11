import numpy as np
import time

def TEST(n, pause=0):
    assert n == 3

    if pause == 0:
        f_list = [lambda x: x[0]**2 + x[1]**2, lambda x: 2 * (x[1] - 2)**2 + x[2]**2]
    else:
        f_list = [lambda x: (time.sleep(pause), x[0]**2 + x[1]**2)[1], lambda x: (time.sleep(pause), 2 * (x[1] - 2)**2 + x[2]**2)[1]] 

    I = [[0, 1], [1, 2]]

    x0 = np.array([1, 0.5, 1])

    return f_list, x0, I

def ARWHEAD(n, pause=0):
    f_list = [] 
    I = []

    for i in range(n-1):
        
        if pause == 0:
            f_list.append(lambda x, i=i: (x[i]**2 + x[n-1]**2)**2 - 4*x[i] + 3)
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), (x[i]**2 + x[n-1]**2)**2 - 4*x[i] + 3)[1])
        I.append([i, n-1])

    assert len(f_list) == len(I) and len(f_list) == n-1

    x0 = np.zeros(n)

    return f_list, x0, I


def BDEXP(n, pause=0):
    f_list = [] 
    I = []

    for i in range(n-2):
        if pause == 0:
            f_list.append(lambda x, i=i: (x[i]+x[i+1]) * np.exp((x[i]+x[i+1])*(-x[i+2])))
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), (x[i]+x[i+1]) * np.exp((x[i]+x[i+1])*(-x[i+2])))[1])
        I.append([i,i+1,i+2])
    
    x0 = np.ones(n)

    return f_list, x0, I


def BDQRTIC(n, pause=0):
    f_list = [] 
    I = []

    for i in range(n-4):
        if pause == 0:
            f_list.append(lambda x, i=i: (x[i]**2 + 2*x[i+1]**2 + 3*x[i+2]**2 + 4*x[i+3]**2 + 5*x[n-1]**2)**2 - 4*x[i] + 3)
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), (x[i]**2 + 2*x[i+1]**2 + 3*x[i+2]**2 + 4*x[i+3]**2 + 5*x[n-1]**2)**2 - 4*x[i] + 3)[1])
        I.append([i,i+1,i+2,i+3, n-1])
    
    x0 = np.ones(n)
    return f_list, x0, I


def BEALES(n, pause=0):
    f_list = [] 
    I = []

    for i in range(int(n/2)):
        j=i*2
        if pause == 0:
            f_list.append(lambda x, j=j: (1.5 - x[j] + x[j] * x[j+1])**2 
                                        + (2.25 - x[j] + x[j] * x[j+1]**2) ** 2 
                                        + (2.625 - x[j] + x[j] * x[j+1]**3) ** 2)
        else:
            f_list.append(lambda x, j=j: (time.sleep(pause), (1.5 - x[j] + x[j] * x[j+1])**2 
                                        + (2.25 - x[j] + x[j] * x[j+1]**2) ** 2 
                                        + (2.625 - x[j] + x[j] * x[j+1]**3) ** 2)[1])
        I.append([j,j+1])
    
    x0 = np.ones(n)

    return f_list, x0, I


def BROWNAL6(n, pause=0):
    f_list = [] 
    I = []
    
    for i in range(int((n-2)/4)):
        j = i * 4 
        f_part = []
        
        for k in range(0,5):
            f_part.append(lambda x, j=j: (x[j+k] + x[j] + x[j+1] + x[j+2] + x[j+3] + x[j+4] + x[j+5] - 7)**2)
        
        if pause == 0:
            f_list.append(lambda x, j=j: f_part[0](x) + f_part[1](x) + f_part[2](x) + f_part[3](x) + f_part[4](x) + (x[j]*x[j+1]*x[j+2]*x[j+3]*x[j+4]*x[j+5]-1) **2)
        else:
            f_list.append(lambda x, j=j: (time.sleep(pause), f_part[0](x) + f_part[1](x) + f_part[2](x) + f_part[3](x) + f_part[4](x)
                                                         + (x[j]*x[j+1]*x[j+2]*x[j+3]*x[j+4]*x[j+5]-1) **2)[1])
        
        
        I.append([j, j + 1, j + 2, j + 3, j + 4, j + 5])

    x0 = 0.5 * np.ones(n)
    
    return f_list, x0, I


def BROYDN3D(n, pause=0):
    
    f_list = [] 
    I = []

    if pause == 0:
        f_list.append(lambda x: ((3 - 2*x[0]) * x[0] - 2* x[1] + 1)**2)    
    else:
        f_list.append(lambda x: (time.sleep(pause), ((3 - 2*x[0]) * x[0] - 2* x[1] + 1)**2)[1])
    I.append([0,1])
    

    for i in range(1, n-1):
            
        if pause == 0:
            f_list.append(lambda x, i=i: ((3 - 2*x[i]) * x[i] - x[i-1] - 2* x[i+1] + 1)**2)
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), ((3 - 2*x[i]) * x[i] - x[i-1] - 2* x[i+1] + 1)**2)[1])
        I.append([i-1, i, i+1])

    if pause == 0:
        f_list.append(lambda x: ((3 - 2*x[n-1]) * x[n-1] - x[n-2] + 1)**2)    
    else:
        f_list.append(lambda x: (time.sleep(pause), ((3 - 2*x[n-1]) * x[n-1] - x[n-2] + 1)**2)[1])
    I.append([n-2, n-1])
    
    x0 = -1 * np.ones(n);

    return f_list, x0, I


def DIXMAANA(n, pause=0):
    f_list = [] 
    I = []

    m=int(n/3);

    for i in range(n):
        if i < m:
            if pause == 0:
                f_list.append(lambda x, i=i: 1 + x[i]**2 + 0.125 * x[i]**2 * x[i+m]**4 + 0.125 * x[i] * x[i + 2 * m])
            else:
                f_list.append(lambda x, i=i: (time.sleep(pause), 1 + x[i]**2 + 0.125 * x[i]**2 * x[i+m]**4 + 0.125 * x[i] * x[i + 2 * m])[1])
            I.append([i, i+m, i+2*m])

        elif i < 2*m:
            if pause == 0:
                f_list.append(lambda x, i=i: 1 + x[i]**2 + 0.125 * x[i]**2 * x[i+m]**4)
            else:
                f_list.append(lambda x, i=i: (time.sleep(pause), 1 + x[i]**2 + 0.125 * x[i]**2 * x[i+m]**4)[1])
            I.append([i, i+m])

        else:
            if pause == 0:
                f_list.append(lambda x, i=i: 1 + x[i]**2)
            else:
                f_list.append(lambda x, i=i: (time.sleep(pause), 1 + x[i]**2)[1])
            I.append([i])
    
    x0=2*np.ones(n)

    return f_list, x0, I


def DIXMAANI(n, pause=0):
    f_list = [] 
    I = []

    m=int(n/3);
    
    for i in range(n):
        if i < m:
            if pause == 0:
                f_list.append(lambda x, i=i: 1 + x[i]**2 *(i/n)**2 + 0.125 * x[i]**2 *x[i+m]**4 + 0.125 * x[i] * x[i+2*m] * (i/n))
            else:
                f_list.append(lambda x, i=i: (time.sleep(pause),1 + x[i]**2 *(i/n)**2 + 0.125 * x[i]**2 *x[i+m]**4 + 0.125 * x[i] * x[i+2*m] * (i/n))[1])
            I.append([i, i+m, i+2*m])

        elif i < 2*m:
            if pause == 0:
                f_list.append(lambda x, i=i: 1 + x[i]**2 *(i/n)**2 +0.125 * x[i]**2  * x[i+m]**4)
            else:
                f_list.append(lambda x, i=i: (time.sleep(pause),1 + x[i]**2 *(i/n)**2 +0.125 * x[i]**2  * x[i+m]**4)[1])
            I.append([i, i+m])

        else:
            if pause == 0:
                f_list.append(lambda x, i=i: 1 + x[i]**2 *(i/n)**2)
            else:
                f_list.append(lambda x, i=i: (time.sleep(pause),1 + x[i]**2 *(i/n)**2)[1])
            I.append([i])
    
    x0=2*np.ones(n)

    return f_list, x0, I


def ENGVAL(n, pause=0):
    f_list = [] 
    I = []

    for i in range(n-1):
        if pause == 0:
            f_list.append(lambda x, i=i: (x[i]**2 + x[i+1]**2)**2 - 4*x[i] + 3.0)
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), (x[i]**2 + x[i+1]**2)**2 - 4*x[i] + 3.0)[1])
        I.append([i, i+1])

    x0 = 2 * np.ones(n)

    return f_list, x0, I


def FREUROTH(n, pause=0):
    f_list = [] 
    I = []
    
    for i in range(n-1):
        if pause == 0:
            f_list.append(lambda x, i=i: (-13 + x[i] + ((5-x[i+1]) * x[i+1] - 2) * x[i+1])**2 + (-29 + x[i] + ((x[i+1]+1) * x[i+1] - 14) * x[i+1])**2)
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), (-13 + x[i] + ((5-x[i+1]) * x[i+1] - 2) * x[i+1])**2 
                                                         + (-29 + x[i] + ((x[i+1]+1) * x[i+1] - 14) * x[i+1])**2)[1])

        I.append([i, i+1])

    x0 = np.zeros(n)
    x0[0] = 0.5
    x0[1] = -2

    return f_list, x0, I


def MOREBV(n, pause=0):
    f_list = []
    I = []
    
    h = 1 / (n+1)

    if pause == 0:
        f_list.append(lambda x, h=h: (2*x[0] - x[1] + h**2 * (x[0] + h + 1)**(3/2))**2)
    else:
        f_list.append(lambda x, h=h: (time.sleep(pause), (2*x[0] - x[1] + h**2 * (x[0] + h + 1)**(3/2))**2)[1])
    I.append([0,1])

    for i in range(1, n-1):
       
        if pause == 0:
            f_list.append(lambda x, i=i, h=h: (2*x[i] - x[i-1] -x[i+1] + h**2 * (x[i] + i * h + 1)**(3/2))**2)
        else:
            f_list.append(lambda x, i=i, h=h: (time.sleep(pause), (2*x[i] - x[i-1] -x[i+1] + h**2 * (x[i] + i * h + 1)**(3/2))**2)[1])
        I.append([i-1, i, 1+1])

    if pause == 0:
        f_list.append(lambda x, h=h: (2*x[n-1] - x[n-2] + h**2 * (x[n-1] + (n-1) * h + 1)**(3/2))**2)
    else:
        f_list.append(lambda x, h=h: (time.sleep(pause), (2*x[n-1] - x[n-2] + h**2 * (x[n-1] + (n-1) * h + 1)**(3/2))**2)[1])
    I.append([n-2,n-1])
    
    x0 = np.zeros(n)
    for i in range(n):
        x0[i] = i * h * (i*h - 1);
    
    return f_list, x0, I


def NZF1(n, pause=0):
    f_list = []
    I = []

    l = n // 13
    
    for i in range(l):
        j = i * 13
        
        if pause == 0:
            f_list.append(lambda x, j=j: (3 * x[j] - 60 + (1/10) * (x[j+1] - x[j+2])**2)**2)
            f_list.append(lambda x, j=j: (x[j+1]**2 + x[j+2]**2 + x[j+3]**2 * (1 + x[j+3]**2) + x[j+6] + x[j+5] / (1 + x[j+4]**2 + np.sin(x[j+4] / 1000)))**2)
            f_list.append(lambda x, j=j: (x[j+6] + x[j+7] - x[j+8]**2 + x[j+10])**2)
            f_list.append(lambda x, j=j: (np.log(1 + x[j+10]**2) + x[j+11] - 5 * x[j+12] + 20)**2)            
            f_list.append(lambda x, j=j: (x[j+4] + x[j+5] + x[j+5] * x[j+9] + 10 * x[j+9] - 50)**2)
        else:
            f_list.append(lambda x, j=j: (time.sleep(pause), (3 * x[j] - 60 + (1/10) * (x[j+1] - x[j+2])**2)**2)[1])
            f_list.append(lambda x, j=j: (time.sleep(pause), (x[j+1]**2 + x[j+2]**2 + x[j+3]**2 * (1 + x[j+3]**2) + x[j+6] + x[j+5] / (1 + x[j+4]**2 + np.sin(x[j+4] / 1000)))**2)[1])
            f_list.append(lambda x, j=j: (time.sleep(pause), (x[j+6] + x[j+7] - x[j+8]**2 + x[j+10])**2)[1])
            f_list.append(lambda x, j=j: (time.sleep(pause), (np.log(1 + x[j+10]**2) + x[j+11] - 5 * x[j+12] + 20)**2)[1])
            f_list.append(lambda x, j=j: (time.sleep(pause), (x[j+4] + x[j+5] + x[j+5] * x[j+9] + 10 * x[j+9] - 50)**2)[1])

        I.append([j, j+1, j+2])
        I.append([j+1, j+2, j+3, j+4, j+5, j+6])
        I.append([j+6, j+7, j+8, j+10])
        I.append([j+10, j+11, j+12])
        I.append([j+4, j+5, j+9])
        
        if l > 0 and i < l-1:
            if pause == 0:
                f_list.append(lambda x, j=j: ((x[j+6] - x[j+19])**2))
            else:
                f_list.append(lambda x, j=j: (time.sleep(pause), ((x[j+6] - x[j+19])**2))[1])
            I.append([j+6, j+19])    
    
    x0 = np.ones(n)
    
    return f_list, x0, I


def POWSING(n, pause=0):

    f_list = []
    I = []
    
    for i in range(int(n/4)):
        j = i * 4 
        if pause == 0:
            f_list.append(lambda x, j=j:  (x[j] + 10*x[j+1])**2 + 5*(x[j+2] - x[j+3])**2 + (x[j+1] - 2*x[j+2])**4 + 10*(x[j] - x[j+3])**4)
        else:
            f_list.append(lambda x, j=j:  (time.sleep(pause), (x[j] + 10*x[j+1])**2 + 5*(x[j+2] - x[j+3])**2 
                                                            + (x[j+1] - 2*x[j+2])**4 + 10*(x[j] - x[j+3])**4)[1])

        I.append([j,j+1,j+2,j+3])
    
    x0 = np.tile([3,-1,0,1], int(n/4))

    return f_list, x0, I


def ROSENBR(n, pause=0):

    f_list = []
    I = []
    
    for i in range(int(n/2)):
        j = 2*i
        if pause == 0:
            f_list.append(lambda x, j=j: 100*(x[j]**2 - x[j+1])**2 +(x[j] -1)**2)
        else:
            f_list.append(lambda x, j=j: (time.sleep(pause), 100*(x[j]**2 - x[j+1])**2 +(x[j] -1)**2)[1])
        I.append([j, j+1])
    
    x0 = np.zeros(n)
    for i in range(int(n/2)):
        j = 2*i
        x0[j] = -1.2
        x0[j+1] = 1.

    return f_list, x0, I


def TRIDIA(n, pause=0):
    f_list = []
    I = []
    
    if pause == 0:
        f_list.append(lambda x: (x[0] - 1)**2)
    else:
        f_list.append(lambda x: (time.sleep(pause), (x[0] - 1)**2)[1])
    I.append([0])
    
    for i in range(1,n):
        if pause == 0:
            f_list.append(lambda x, i=i: (i * (-x[i-1] + 2 *x[i])**2))
        else:
            f_list.append(lambda x, i=i: (time.sleep(pause), (i * (-x[i-1] + 2 *x[i])**2))[1])
        I.append([i-1, i])
    
    x0 = np.ones(n)

    return f_list, x0, I


def WOODS_NEW(n, pause=0):

    f_list = []
    I = []

    i=0
    while i <= (n - 4):
        if pause == 0:
            f_list.append(lambda x, i=i: (10 * (x[i+1] - x[i]**2))**2)
            f_list.append(lambda x, i=i: (1 - x[i])**2)
            f_list.append(lambda x, i=i: (np.sqrt(90) * (x[i+3] - x[i+2]**2))**2)
            f_list.append(lambda x, i=i: (1 - x[i+2])**2)
            f_list.append(lambda x, i=i: (np.sqrt(10) * (x[i+1] + x[i+3] - 2))**2)
            f_list.append(lambda x, i=i: ((1 / np.sqrt(10)) * (x[i+1] - x[i+3]))**2)

        else:
            f_list.append(lambda x, i=i: (time.sleep(pause),(10 * (x[i+1] - x[i]**2))**2)[1])
            f_list.append(lambda x, i=i: (time.sleep(pause),(1 - x[i])**2)[1])
            f_list.append(lambda x, i=i: (time.sleep(pause),(np.sqrt(90) * (x[i+3] - x[i+2]**2))**2)[1])
            f_list.append(lambda x, i=i: (time.sleep(pause),(1 - x[i+2])**2)[1])
            f_list.append(lambda x, i=i: (time.sleep(pause),(np.sqrt(10) * (x[i+1] + x[i+3] - 2))**2)[1])
            f_list.append(lambda x, i=i: (time.sleep(pause),((1 / np.sqrt(10)) * (x[i+1] - x[i+3]))**2)[1])
        
        I.append([i, i+1])
        I.append([i])
        I.append([i+2, i+3])
        I.append([i+2])
        I.append([i+1, i+3])
        I.append([i+1, i+3])
        
        i += 4
    
    x0 = np.array([-3 if i % 2 == 0 else -1 for i in range(n)])
    
    return f_list, x0, I



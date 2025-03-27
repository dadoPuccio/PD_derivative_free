import numpy as np
import time

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

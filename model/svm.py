from libsvm import *


def cross_validation(x,y):
    #hyparameter: C, gamma

    bestcv = 0
    for log2c in range(-1, 4):
        for log2g in range(-4, 1):
            cmd = f'-v 5 -c {2**log2c} -g {2**log2g} -m 300'
            cv = svm_train(y, x, cmd)
            if cv >= bestcv:
                bestcv = cv
                bestc = 2 ** log2c
                bestg = 2 ** log2g

    return bestc, bestg




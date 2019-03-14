import math


def depth(dim, size=64):
    return math.ceil(math.log(dim/size)/math.log(2))

def leaf_size(dim, depth):
    return dim/2**depth

def subregions(depth):
    s = 0
    for i in reversed(range(0, depth)):
        s += (i+1)*2**i
    return s

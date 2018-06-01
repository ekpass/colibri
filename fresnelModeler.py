from math import hypot, ceil
import numpy as np
from math import pi, cos, tan
from scipy.special import jv  # Bessel function
from itertools import product


class memoize:

    """ Memoization decorator to cache repeatedly-used function calls """

    # stock code from http://avinashv.net

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


@memoize
def lommel(n, a, b):

    """ Calculates the nth lommel function """

    U = 0
    for k in xrange(0, 100000):
        sum = ((-1)**k * (a/b)**(n+2*k) * jv(n+2*k, pi*a*b))
        U += sum
        if abs(sum) < 0.00001:
            return U
    raise ValueError("Failure to converge")


@memoize
def generatePoints(starR):

    """ Models star as an array of uniformly distributed point sources """

    if starR == 0:  # model as point source
        return np.array([(0,0)])
    n = 5  # number of points to model 1D radius of star
    pairs = np.array([item for item in product(np.linspace(-starR, starR, 2*n-1), repeat=2) if hypot(item[0], item[1]) <= starR])
    return pairs


def diffractionCalc(r, p, starR, lam, D, b):

    """ Analytically calculates intensity at a given distance from star centre """

    # r is distance between line of sight and centre of the disk, in fresnel scale units
    # p is radius of KBO, in fresnel scale units
    pts = generatePoints(starR)
    r = fresnel(r, lam, D)
    res = 0
    effR = np.round(np.hypot((r - pts[:, 0]), (pts[:, 1]-b)), 2)
    coslist = np.cos(0.5*pi*(effR**2 + p**2))
    sinlist = np.sin(0.5*pi*(effR**2 + p**2))
    l = len(pts)
    for n in xrange(0, l):
        if effR[n] > p:
            U1 = lommel(1, p, effR[n])
            U2 = lommel(2, p, effR[n])
            res += (1 + (U2 ** 2) + (U1 ** 2) + 2*(U2*coslist[n] - U1*sinlist[n]))
        elif effR[n] == p:
            res += (0.25 * ((jv(0, pi*p*p)**2) + 2*cos(pi*p*p)*jv(0, pi*p*p) + 1))
        else:
            res += ((lommel(0, effR[n], p)**2) + (lommel(1, effR[n], p) ** 2))
    return res / l


def fresnel(x, lam, D):

    """ Converts value to fresnel scale units """
    return x / (lam*D/2.)**(1/2.)


def generateKernel(lam, objectR, b, D, starR):

    """ Calculates the light curve at a given wavelength """

    p = fresnel(objectR, lam, D)  # converting KBO radius to Fresnel Scale
    s = fresnel(starR, lam, D)  # converting effective star radius to Fresnel Scale
    b = fresnel(b, lam, D)
    r = 10000.  # distance between line of sight and centre of the disk in m
    z = [diffractionCalc(j, p, s, lam, D, b) for j in np.arange(-r, r, 10)]
    return z


def defineParam(startLam, endLam, objectR, b, D, angDi):

    """ Simulates light curve for given parameters """

    # startLam: start of wavelength range, m
    # endLam: end of wavelength range, m
    # objectR: radius of KBO, m
    # b: impact parameter, m
    # D: distance from KBO to observer, in AU
    # angDi: angular diameter of star, mas
    # Y: light profile during diffraction event

    D *= 1.496e+11 # converting to metres
    starR = effStarRad(angDi, D)
    n = 20
    if endLam == startLam:
        Y = generateKernel(startLam, objectR, b, D, starR)
    else:
        step = (endLam-startLam) / n
        Y = np.array([generateKernel(lam, objectR, b, D, starR) for lam in np.arange(startLam, endLam, step)])
        Y = np.sum(Y, axis=0)
        Y /= n
    return Y


def effStarRad(angDi, D):

    """ Determines projected star radius at KBO distance """

    angDi /= 206265000.  # convert to radians
    return D * tan(angDi / 2.)


def vT(a, vE):

    """ Calculates transverse velocity of KBO """

    # a is distance to KBO, in AU
    # vE is Earth's orbital speed, in m/s
    # returns vT, transverse KBO velocity, in m/s

    return vE * ( 1 - (1./a)**(1/2.))


def integrateCurve(exposure, curve, totTime, shiftAdj):

    """ Reduces resolution of simulated light curve to match what would be observed for a given exposure time """

    timePerFrame = totTime / len(curve)
    numFrames = roundOdd(exposure/timePerFrame)
    if shiftAdj < 0:
        shiftAdj += 1
    shift = ((len(curve) / 2)% numFrames) - (numFrames-1)/2
    while shift < 0:
        shift += numFrames
    shift += int(numFrames*shiftAdj)
    for index in np.arange((numFrames-1)/2 + shift, len(curve)-(numFrames-1)/2, numFrames):
        indices = range(index - (numFrames-1)/2, index+1+(numFrames-1)/2)
        av = np.average(curve[indices])
        curve[indices] = av
    last = indices[-1]+1  # bins leftover if light curve length isn't divisible by exposure time
    curve[last:] = np.average(curve[last:])
    curve[:shift] = np.average(curve[:shift])
    return curve, numFrames


def roundOdd(x):

    """ Rounds x to the nearest odd integer """

    x = ceil(x)
    if x % 2 == 0:
        return int(x-1)
    return int(x)


def genCurve(exposure, startLam, endLam, objectRad, impact, dist, angDi, shiftAdj):

    """ Convert diffraction pattern to time series """

    velT = vT(dist, 29800)
    curve = defineParam(startLam, endLam, objectRad, impact, dist, angDi)
    n = len(curve)*10./velT
    curve, num = integrateCurve(exposure, curve, n, shiftAdj)
    return curve[::num]


kernels = [genCurve(0.05991, 4e-7, 7e-7, objR, imp, 40, angDi, shiftAdj) for angDi in [0.01, 0.03, 0.08] for objR in [750, 1375, 2500] for shiftAdj in [0, 0.25, 0.5] for imp in [0, objR]]

np.savetxt('kernelstest.txt', kernels)

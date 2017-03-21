import sep
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.convolution import convolve_fft, MexicanHat1DKernel
from joblib import delayed, Parallel
from copy import deepcopy
import multiprocessing
import time
import datetime
import numpy.ma as ma
import os
import struct
import gc
import wcsget  # adaptation of Astrometry.net API


def getNum(f):
    """ Extract Unix time from .vid filename """

    num = f.split('_')[1]
    return int(num, 16)


def diffMatch(template, data, unc):
    """ Calculates the best start position (minX) and the minimization constant associated with this match (minChi) """

    minChi = np.inf
    minX = np.inf
    for st in xrange(((len(data) / 2) - len(template) / 2) - 3, (len(data) / 2) - (len(template) / 2) + 3):
        sum = 0
        for val in xrange(0, len(template)):
            sum += (abs(template[val] - data[st + val])) / abs(unc[st + val])
        if sum < minChi:
            minChi = sum
            minX = st
    return minChi, minX


def initialFind(data, img, img2):
    """ Locates the stars in the initial time slice """

    """ Background extraction for initial time slice"""
    data_new = deepcopy(data)
    data_new[1:, :] /= img[1:, :]
    data_new -= img2
    bkg = sep.Background(data_new)
    bkg.subfrom(data_new)
    thresh = 3. * bkg.globalrms  # set detection threshold to mean + 3 sigma
    """ Identify stars in initial time slice """
    objects = sep.extract(data_new, thresh)

    """ Characterize light profile of each star"""
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    """ Generate tuple of (x,y) positions for each star"""
    positions = zip(objects['x'], objects['y'])

    return positions, halfLightRad, thresh


def refineCentroid(data, coords, sigma):
    """ Refines the centroid for each star for a set of test slices of the data cube """

    xInit = [pos[0] for pos in coords]
    yInit = [pos[1] for pos in coords]
    new_pos = np.array(sep.winpos(data, xInit, yInit, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    return zip(x, y)


def averageDrift(positions, end, frames):
    """ Determines the per-frame drift rate of each star """

    xDrift = np.array([np.subtract(positions[t, :, 0], positions[t - 1, :, 0]) for t in range(1, end)])
    yDrift = np.array([np.subtract(positions[t, :, 1], positions[t - 1, :, 1]) for t in range(1, end)])
    return np.median(xDrift[1:], 0) / frames, np.median(yDrift[1:], 0) / frames


def timeEvolve(data, coords, xDrift, yDrift, r, stars, xTot, yTot, img, img2, t):
    """ Adjusts aperture based on star drift and calculates flux in aperture"""

    x = [coords[ind, 0] + xDrift[ind] for ind in range(0, stars)]
    y = [coords[ind, 1] + yDrift[ind] for ind in range(0, stars)]
    inds = clipCutStars(x, y, xTot, yTot)
    inds = list(set(inds))
    inds.sort()
    xClip = np.delete(np.array(x), inds)
    yClip = np.delete(np.array(y), inds)
    data[1:, :] /= img[1:, :]
    data -= img2
    bkg = np.median(data) * np.pi * r * r
    fluxes = (sep.sum_circle(data, xClip, yClip, r)[0] - bkg).tolist()
    for a in inds:
        fluxes.insert(a, 0)
    coords = zip(x, y, fluxes, [t] * len(x))
    return coords


def clipCutStars(x, y, xTot, yTot):
    """ When the aperture is near the edge of the field of view, sets flux to zero to prevent fadeout"""

    r = 20.
    xeff = np.array(x)
    yeff = np.array(y)
    ind = np.where(r > xeff)
    ind = np.append(ind, np.where(xeff >= (xTot - r)))
    ind = np.append(ind, np.where(r > yeff))
    ind = np.append(ind, np.where(yeff >= (yTot - r)))
    return ind


def kernelDetection(fluxProfile, kernel, kernels, num):
    """ Detects dimming using Mexican Hat kernel for dip detection and set of Fresnel kernels for kernel matching """

    """ Prunes profiles"""
    light_curve = np.trim_zeros(fluxProfile[1:])
    if len(light_curve) == 0:
        return -2  # reject empty profiles
    med = np.median(light_curve)
    indices = np.where(light_curve > min(med * 2, med + 5 * np.std(light_curve)))
    light_curve = np.delete(light_curve, indices)
    trunc_profile = np.where(light_curve < 0, 0, light_curve)
    if len(trunc_profile) < 1100:
        return -2  # reject objects that go out of frame rapidly, ensuring adequate evaluation of median flux
    if abs(np.mean(trunc_profile[:1000]) - np.mean(trunc_profile[-1000:])) > np.std(trunc_profile[:1000]):
        return -2  # reject tracking failures
    if np.median(trunc_profile) < 5000:
        return -2  # reject stars that are very dim, as SNR is too poor
    m = np.std(trunc_profile[900:1100]) / np.median(trunc_profile[900:1100])

    """ Dip detection"""
    conv = convolve_fft(trunc_profile, kernel)
    minPeak = np.argmin(conv)
    minVal = min(conv)
    if 30 <= minPeak < len(trunc_profile) - 30:
        med = np.median(
            np.concatenate((trunc_profile[minPeak - 100:minPeak - 30], trunc_profile[minPeak + 30:minPeak + 100])))
        trunc_profile = trunc_profile[(minPeak - 30):(minPeak + 30)]
        unc = ((np.sqrt(abs(trunc_profile) / 100.) / np.median(trunc_profile)) * 100)
        unc = np.where(unc == 0, 0.01, unc)
        trunc_profile /= med
    else:
        return -2  # reject events that are cut off at the start/end of time series
    bkgZone = conv[10: -10]

    if minVal < np.mean(bkgZone) - 3.75 * np.std(bkgZone):  # dip detection threshold
        """ Kernel matching"""
        old_min = np.inf
        for ind in range(0, len(kernels)):
            if min(kernels[ind]) > 0.8:
                continue
            new_min, loc = diffMatch(kernels[ind], trunc_profile, unc)
            if new_min < old_min:
                active_kernel = ind
                min_loc = loc
                old_min = new_min
        unc_l = unc[min_loc:min_loc + len(kernels[active_kernel])]
        thresh = 0
        for u in unc_l:
            thresh += (abs(m) ** 1) / (abs(u) ** 1)  # kernel match threshold
        if old_min < thresh:
            critTime = np.where(fluxProfile == light_curve[minPeak])[0]
            print datetime.datetime.now(), "Detected candidate: frame", str(critTime) + ", star", num
            if len(critTime) > 1:
                raise ValueError
            return critTime[0]  # returns location in original time series where dip occurs
        else:
            return -1  # reject events that do not pass kernel matching
    else:
        return -1  # reject events that do not pass dip detection


def readByte(file, start, length):
    """ Returns the integer located at a given position in the file """

    file.seek(start)
    byte = file.read(length)
    return struct.unpack('i', byte + (b'\0' * (4 - len(byte))))[0]


def getSize(filename, iterations):
    """ Imports file from .vid format """

    with open(filename, "rb") as file:
        magic = readByte(file, 0, 4)
        if magic != 809789782:
            print datetime.datetime.now(), "Error: invalid .vid file"
            return
        totBytes = os.stat(filename).st_size  # size of file in bytes
        seqlen = readByte(file, 4, 4)  # size of image data in bytes
        width = readByte(file, 30, 2)  # image width
        height = readByte(file, 32, 2)  # image height
        frames = totBytes // seqlen  # number of frames in the image
        bytesToNum = readByte(file, 34, 2) // 8  # number of bytes per piece of data
        if frames % 2 == 0:
            priorByte = totBytes / 2 * iterations
            frames /= 2
        else:
            if iterations == 0:
                priorByte = 0
                frames = (frames - 1) / 2
            else:
                frames = (frames + 1) / 2
                priorByte = (frames - 1) * bytesToNum

        file.seek(priorByte, os.SEEK_SET)
        x = np.fromfile(file, dtype='int32', count=width * height * frames / 2)
        unixTime = x[5::width * height / 2.]  # get seconds since epoch
        micro = x[6::width * height / 2] / 1000000.  # get additional milliseconds
        timeList = [np.float(z) + np.float(y) for y, z in zip(unixTime, micro)]

    return width, height, frames, timeList


def importFrames(filename, iterations, frameNum, length):
    """ Imports file from .vid format """

    with open(filename, "rb") as file:
        magic = readByte(file, 0, 4)
        if magic != 809789782:
            print datetime.datetime.now(), "Error: invalid .vid file"
            return
        totBytes = os.stat(filename).st_size  # size of file in bytes
        seqlen = readByte(file, 4, 4)  # size of image data in bytes
        headlen = readByte(file, 8, 4)  # size of header data in bytes
        width = readByte(file, 30, 2)  # image width
        height = readByte(file, 32, 2)  # image height
        bytesToNum = readByte(file, 34, 2) // 8  # number of bytes per piece of data
        frames = totBytes // seqlen  # number of frames in the image
        area = width * height
        headerData = headlen // bytesToNum
        if frames % 2 == 0:
            priorByte = totBytes / 2 * iterations
        else:
            if iterations == 0:
                priorByte = 0
            else:
                frames = (frames + 1) / 2
                priorByte = (frames - 1) * bytesToNum
        file.seek(priorByte + width * height * bytesToNum * frameNum, os.SEEK_SET)
        c = width * height * length
        if c < 0:
            c = -1
        x = np.fromfile(file, dtype='uint16', count=c)
        for headerCell in range(0, headerData):
            x[headerCell::area] = 0
    x = np.reshape(x, [-1, width, height])
    if x.shape[0] == 1:
        x = x[0]
        x = x.astype('float64')
    return x


def camCompare(ind, results, positions, nameStamp, dayStamp, directory):
    """ Saves data for later comparison between EMCCD1 and EMCCD2 using compareStreams.py """

    ''' select only the xy coordinates of time series flagged as containing occultation event '''
    flaggedPos = positions[:, ind, :]
    shp = flaggedPos[:, :, 0].shape[:2]
    xs = flaggedPos[0, :, 0]
    ys = flaggedPos[0, :, 1]

    ''' convert xy coordinates to FITS binary table format'''
    c1 = fits.Column(name='XIMAGE', format='D', array=xs)
    c2 = fits.Column(name='YIMAGE', format='D', array=ys)
    tbhdu = fits.BinTableHDU.from_columns([c1, c2])
    tbhdu.writeto("xycoords.xyls", clobber=True)

    ''' convert xy coordinates to RA dec using command line astrometry.net package '''
    os.system('wcs-xy2rd -i xycoords.xyls -o rdcoords.rdls -X XIMAGE -Y YIMAGE -w wcs.fits')

    '''convert resulting rdls file to slices in numpy ndarray'''
    hdulist = fits.open('rdcoords.rdls')
    rds = np.array(hdulist[1].data, dtype=tuple)
    xs = np.array([val[0] for val in rds], dtype=np.float64)
    ys = np.array([val[1] for val in rds], dtype=np.float64)
    xs = np.repeat(xs, shp[0])
    ys = np.repeat(ys, shp[0])
    xs = np.reshape(xs, shp)
    ys = np.reshape(ys, shp)
    flaggedPos = np.dstack((xs, ys, flaggedPos[:, :, 0], flaggedPos[:, :, 1], flaggedPos[:, :, 2], flaggedPos[:, :, 3]))

    ''' shorten time series to focus around occultation event '''
    centres = results[np.where(results > 0)]
    centredPos = np.empty([60, len(centres)], dtype=(np.float64, 6))
    for n in range(0, len(centres)):
        centredPos[:, n] = flaggedPos[centres[n] - 30:centres[n] + 30, n, :]
    np.save(str(dayStamp) + "/" + nameStamp + ".npy", centredPos)  # save x,y,RA,dec,flux,time series
    dt = np.dtype(
        [('starID', np.str, 50), ('RA', np.float64), ('dec', np.float64), ('fluxes', tuple), ('time', np.float64)])
    savedVals = np.empty([len(ind)], dtype=dt)
    for v in range(0, len(ind)):
        savedVals[v] = (
        nameStamp + "-" + str(ind[v]), centredPos[0, v, 0], centredPos[0, v, 1], tuple(centredPos[:, v, 4]),
        centredPos[30, v, 5])

    ''' save data for each camera for later comparison '''
    if directory == '/emccd1/':
        try:
            stream1 = np.load('temp1.npy')
            stream1 = np.append(stream1, savedVals)
        except IOError:
            stream1 = savedVals
        np.save('temp1.npy', stream1)  # if first camera, save matches
        print datetime.datetime.now(), "Directory:", directory
    elif directory == '/emccd2/':
        try:
            stream2 = np.load('temp2.npy')
            stream2 = np.append(stream2, savedVals)
        except IOError:
            stream2 = savedVals
        np.save('temp2.npy', stream2)  # if first camera, save matches
        print datetime.datetime.now(), "Directory:", directory
    else:
        print "Warning: directory not found"


def main(file, half, directory):
    """ Detect possible occultation events in selected file and archive results """

    print datetime.datetime.now(), "Opening:", file, half

    """ Create folder for results"""
    dayStamp = datetime.date.today()
    if not os.path.exists(str(dayStamp)):
        os.makedirs(str(dayStamp))

    """Adjustable parameters"""
    r = 5.  # radius of aperture for flux measurements
    expectedLength = 0.15
    refreshRate = 2.  # number of seconds (as float) between centroid refinements

    """Initialize variables"""
    filename = directory + file + ".vid"

    xTot, yTot, tTot, timeList = getSize(filename, half)
    print datetime.datetime.now(), "Imported", tTot, "frames"
    if tTot < 500:
        print datetime.datetime.now(), "Insufficient length data cube, skipping..."
        return

    if directory == "/emccd1/":  # preprocessing with EMCCD1 corrections
        img = np.loadtxt('flat1.txt', dtype='float64')
        img /= np.mean(img)
        img2 = np.loadtxt('dark1.txt', dtype='float64')
    else:  # preprocessing with EMCCD2 corrections
        img = np.loadtxt('flat2.txt', dtype='float64')
        img /= np.mean(img)
        img2 = np.loadtxt('dark2.txt', dtype='float64')

    timeInt = 0.05991  # exposure length
    kernels = np.loadtxt('kernels_60.txt')

    kernelFrames = int(round(expectedLength / timeInt))
    kernel = MexicanHat1DKernel(kernelFrames)
    evolutionFrames = int(round(refreshRate / timeInt))  # determines the number of frames in X seconds of data

    """Preliminary data processing and star identification"""
    dataZero = importFrames(filename, half, 0, 1)
    initial_positions, radii, detectThresh = initialFind(dataZero, img, img2)
    radii = np.mean(radii * 2. / 2.35)  # calculate gaussian sigma for each star's light profile
    starNum = len(initial_positions)  # determine number of stars in image
    if starNum == 0:
        print datetime.datetime.now(), "No detected stars in image, skipping..."
        return

    positions = np.empty([tTot, starNum], dtype=(np.float64, 4))
    test_pos = np.empty([tTot // evolutionFrames, starNum], dtype=(np.float64, 2))
    cores = multiprocessing.cpu_count()  # determine number of CPUs for parallel processing
    """Time evolution of star"""

    test_pos[0] = refineCentroid(dataZero, initial_positions, radii)
    tFrames = 110
    if test_pos.shape[0] < 110:
        tFrames = test_pos.shape[0]
    for t in range(1, tFrames):
        test_pos[t] = refineCentroid(importFrames(filename, half, t * evolutionFrames, 1), deepcopy(test_pos[t - 1]),
                                     radii)
    xDrift, yDrift = averageDrift(test_pos, tFrames, evolutionFrames)
    dataZero[1:, :] /= img[1:, :]
    dataZero -= img2
    PrimaryHDU(dataZero).writeto("temp.fits", clobber=True)
    subid = wcsget.imageUpload("temp.fits")
    bkg = np.median(dataZero) * np.pi * r * r
    positions[0] = zip(test_pos[0, :, 0], test_pos[0, :, 1],
                       (sep.sum_circle(dataZero, test_pos[0, :, 0], test_pos[0, :, 1], r)[0] - bkg).tolist(),
                       np.ones_like(test_pos[0, :, 0]) * timeList[0])
    for t in range(1, tTot):
        positions[t] = timeEvolve(importFrames(filename, half, t, 1), deepcopy(positions[t - 1]), xDrift, yDrift, r,
                                  starNum, xTot, yTot, img, img2, timeList[t])
    print datetime.datetime.now(), positions.shape

    nameStamp = str(file) + str(half)
    results = np.array(Parallel(n_jobs=cores, backend='threading')(
        delayed(kernelDetection)(positions[:, index, 2], kernel, kernels, index) for index in
        range(0, starNum)))  # perform dip detection and kernel match for all time series
    saveTimes = results[np.where(results > 0)]
    saveChunk = int(round(5 / timeInt))

    for t in saveTimes:  # save data surrounding candidate event
        if t - saveChunk >= 0:  # if chunk does not include lower data boundary
            if t + saveChunk <= tTot:  # if chunk does not include upper data boundary
                np.save("Surrounding-" + nameStamp + "-" + str(np.where(results == t)[0][0]) + ".npy",
                        importFrames(filename, half, t - saveChunk, 1 + saveChunk * 2))
            else:  # if chunk includes upper data boundary, stop at upper boundary
                np.save("Surrounding-" + nameStamp + "-" + str(
                    np.where(results == t)[0][0]) + ".npy""Surrounding-" + nameStamp + "-" + str(
                    np.where(results == t)[0][0]) + ".npy",
                        importFrames(filename, half, t - saveChunk, tTot - t + saveChunk))
        else:  # if chunk includes lower data boundary, start at 0
            np.save("Surrounding-" + nameStamp + "-" + str(np.where(results == t)[0][0]) + ".npy",
                    importFrames(filename, half, 0, t + saveChunk))

    print datetime.datetime.now(), saveTimes
    count = len(np.where(results == -2)[0])
    print datetime.datetime.now(), "Rejected Stars: " + str(round(count * 100. / starNum, 2)) + "%"

    ind = np.where(results > 0)[0]

    if len(ind) > 0:  # if any events detected

        ''' retrieve WCS file from astrometry.net servers'''
        print datetime.datetime.now(), "Fetching WCS from astrometry.net..."
        fileSamp = " "
        while fileSamp != "SIMPL":
            stat = wcsget.fetchWCS(subid)
            if stat == "failure":
                print datetime.datetime.now(), "Insufficient data for plate match, skipping..."
                return
            with open("wcs.fits") as f:
                fileSamp = f.read(5)
                if fileSamp != "SIMPL":  # if WCS download unsuccessful
                    print datetime.datetime.now(), "Astrometry.net job still pending, retrying..."
                    time.sleep(30)
                else:
                    print datetime.datetime.now(), "WCS obtained"

        camCompare(ind, results, positions, nameStamp, dayStamp, directory)
    else:
        print datetime.datetime.now(), "No events detected"

    print datetime.datetime.now(), "Total stars in file:", starNum
    print datetime.datetime.now(), "Candidate events in file:", len(ind)
    print datetime.datetime.now(), "Closing:", file, half
    print "\n"


''' process backlog '''

directoryA = '/emccd1/'  # directory with EMCCD1 data
directoryB = '/emccd2/'  # directory with EMCCD2 data

fileListA = glob(directoryA + '*.vid')
fileListB = glob(directoryB + '*.vid')

fileListA.sort(key=getNum)  # order files from oldest to newest
fileListB.sort(key=getNum)

for x in range(0, len(fileListA)):
    fileListA[x] = fileListA[x][len(directoryA):-4]  # remove non-identifying information in file name
for x in range(0, len(fileListB)):
    fileListB[x] = fileListB[x][len(directoryB):-4]

for f in range(0, len(fileListA)):
    for half in range(0, 2):
        main(fileListA[f], half, directoryA)  # run pipeline for each file in the EMCCD1 directory
        gc.collect()
        main(fileListB[f], half, directoryB)  # run pipeline for each file in the EMCCD2 directory
        gc.collect()

""" once backlog data is processed, continuously check for new data """

while True:
    fileList2A = glob(directoryA + '*vid')
    fileList2B = glob(directoryB + '*.vid')
    fileList2A.sort(key=getNum)
    fileList2B.sort(key=getNum)

    for x in range(0, len(fileList2A)):
        fileList2A[x] = fileList2A[x][len(directoryA):-4]
    for x in range(0, len(fileList2B)):
        fileList2B[x] = fileList2B[x][len(directoryB):-4]

    if set(fileList2A).issubset(set(fileListA)) is False:
        diffA = list(set(fileList2A).difference(set(fileListA)))  # get list of new EMCCD1 data
        diffB = list(set(fileList2B).difference(set(fileListB)))  # get list of new EMCCD2 data
        diffA.sort(key=getNum)
        diffB.sort(key=getNum)
        upper = max((len(diffA), len(diffB)))
        for f in range(0, upper):
            for half in range(0, 2):
                try:
                    main(diffA[f], half, directoryA)  # process new EMCCD1 data
                except IndexError:  # handles uneven timing between the two cameras
                    pass
                gc.collect()
                try:
                    main(diffB[f], half, directoryB)  # process new EMCCD2 data
                except IndexError:
                    pass
                gc.collect()
        fileListA = fileList2A  # mark that all of fileList2A has been processed
        fileListB = fileList2B
    else:
        print datetime.datetime.now(), "No new files"
        time.sleep(3600)  # if no new files, wait an hour before checking again

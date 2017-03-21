import numpy as np


def matchEvents(stream1, stream2):

    """ Compare data from EMCCD1 and EMCCD2 """

    times1 = [event[4] for event in stream1]
    times2 = [event[4] for event in stream2]
    matches = []
    for ind in range(0, len(stream1)):
        minVal = min(times2, key=lambda x: abs(x-times1[ind]))
        if abs(minVal-times1[ind]) < 0.15:  # if events at same time in both streams
            min2 = np.where(np.array(times2) == minVal)[0][0]
            min2 = stream2[min2]
            min1 = stream1[ind]
            if abs(min1[1]-min2[1]) < 0.25 and abs(min1[2]-min2[2]) < 0.25:  # if same star in both streams
                matches.extend([min1, min2])
    return matches

stream1 = np.load('temp1.npy')  # results from mainColibri.py for EMCCD1
stream2 = np.load('temp2.npy')  # results from mainColibri.py for EMCCD2

dt = np.dtype([('starID', np.str, 50), ('RA', np.float64), ('dec', np.float64), ('fluxes', object), ('time', np.float64)])

m = matchEvents(stream1, stream2)
print len(m)/2  # print number of matches

returnArray = np.empty([len(m)], dtype=dt)
for val in range(0, len(m)):
    returnArray[val] = tuple(m[val])

np.save('matches.npy', returnArray)

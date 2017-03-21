
""" These helper functions are an adapted form of the Astrometry.net API.  For further details on this code,
    see the original:  https://github.com/dstndstn/astrometry.net/blob/master/net/client/client.py """

from __future__ import print_function
import requests
import json
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email.encoders import encode_noop
from cStringIO import StringIO
from email.generator import Generator
from urllib2 import Request, urlopen, HTTPError, URLError
import time
import urllib


def getNum(f):
    name = f.partition('.')[0]
    alpha, num = name.split('-')
    return int(num)


def json2python(data):
    try:
        return json.loads(data)
    except:
        pass
    return None


def imageUpload(image):

    python2json = json.dumps

    R = requests.post('http://nova.astrometry.net/api/login', data={'request-json': json.dumps({"apikey": "INSERTKEYNAMEHERE"})})
    Rdata = R.text
    try:
        Rdata = json.loads(Rdata)
    except ValueError as e:
        print(datetime.datetime.now(), "ConnectionError", e)
        print(datetime.datetime.now(), "Astrometry.net may be down, reattempting upload...")
        time.sleep(5)
        return imageUpload(image)
    
    session = (Rdata['session'])
    json1 = python2json({'session': session})

    f = open(image, 'rb')
    m1 = MIMEBase('text', 'plain')
    m1.add_header('Content-disposition', 'form-data; name="request-json"')
    m1.set_payload(json1)

    m2 = MIMEApplication(f.read(),'octet-stream', encode_noop)
    m2.add_header('Content-disposition', 'form-data; name="file"; filename="%s"' % image)
    mp = MIMEMultipart('form-data', None, [m1, m2])

    class MyGenerator(Generator):
        def __init__(self, fp, root=True):
            Generator.__init__(self, fp, mangle_from_=False,
                               maxheaderlen=0)
            self.root = root

        def _write_headers(self, msg):
            if self.root:
                return
            for h, v in msg.items():
                print(('%s: %s\r\n' % (h,v)), end='', file=self._fp)
            print('\r\n', end='', file=self._fp)

        def clone(self, fp):
            return MyGenerator(fp, root=False)

    fp = StringIO()
    g = MyGenerator(fp)
    g.flatten(mp)
    data = fp.getvalue()
    headers = {'Content-type': mp.get('Content-type')}
    url = 'http://nova.astrometry.net/api/upload'
    request = Request(url=url, headers=headers, data=data)
    print(datetime.datetime.now(), "Attempting upload...")
    try:
        f = urlopen(request)
        txt = f.read()
        result = json2python(txt)
        stat = result.get('status')
        print(datetime.datetime.now(), 'Upload to Astrometry.net:', stat)
        return result.get('subid')

    except HTTPError as e:
        print(datetime.datetime.now(), 'HTTPError', e)
        print(datetime.datetime.now(), 'Reattempting upload...')
        return imageUpload(image)

    except URLError as e:
        print(datetime.datetime.now(), 'URLError', e)
        print(datetime.datetime.now(), 'Reattempting upload...')
        return imageUpload(image)


def fetchWCS(subid):

    R = requests.post('http://nova.astrometry.net/api/submissions/'+str(subid), data={'request-json': json.dumps({"apikey": "INSERTKEYNAMEHERE"})})
    Rdata = R.text
    try:
        Rdata = json.loads(Rdata)
    except ValueError as e:
        print(datetime.datetime.now(), "ConnectionError", e)
        print(datetime.datetime.now(), "Astrometry.net may be down, reattempting retrieval...")
        time.sleep(5)
        return fetchWCS(subid)
    try:
        id = Rdata['jobs'][0]
    except IndexError:
        id = None
    while id is None:
        time.sleep(10)
        R = requests.post('http://nova.astrometry.net/api/submissions/' + str(subid), data={'request-json': json.dumps({"apikey": "INSERTKEYNAMEHERE"})})
        Rdata = R.text
        Rdata = json.loads(Rdata)
        try:
            id = Rdata['jobs'][0]
        except IndexError:
            id = None
        print(datetime.datetime.now(), "File processing...")
    R = requests.post('http://nova.astrometry.net/api/jobs/'+str(id), data={'request-json': json.dumps({"apikey": "INSERTKEYNAMEHERE"})})
    Rdata = R.text
    try:
        Rdata = json.loads(Rdata)
    except ValueError as e:
        print(datetime.datetime.now(), "ConnectionError", e)
        print(datetime.datetime.now(), "Astrometry.net may be down, reattempting retrieval...")
        time.sleep(5)
        return fetchWCS(subid)
    stat = Rdata['status']
    urllib.urlretrieve('http://nova.astrometry.net/wcs_file/'+str(id), "wcs.fits")
    return stat

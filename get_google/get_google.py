#!/usr/bin/env python
#encoding:utf-8
import sys,os,re
import json
import httplib
import contextlib
import urlparse
import base64
import time
import hashlib
import urllib
import urllib2

def getsimi( objurl, outputfile ):
    f = open(outputfile,'w')
    print >> f, objurl
    hosturl="https://www.google.com/searchbyimage"
    address = urlparse.urlparse(hosturl)
    req = urllib.urlencode(
        {
                #'image_url': "http://img3.xcarimg.com/drive/14231/16413/608_20150816194043599581701224907.jpg",
                'image_url': objurl,
                'btnG': '按图片搜索',
                'image_content': "" ,
                'filename': ""
        } )
    print >> sys.stderr,req
    headers = {
        'Accept': 'text/html,,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'accept-language':'zh-CN,zh;q=0.8',
        'cookie':'PREF=ID=1111111111111111:FF=0:TM=1439880102:LM=1439880102:V=1:S=ZhoEmk5zL4i2optM; NID=70=SboTlroPenZJsh0CzuFo2fipNuwYdUVdxlTlNFu16DbvuhJL-Kx2NphcShsrkpd3BbJn6Trd9TqKNxPBNoOEWCKGbqrVP3Rn_l9kgCofRRp6hj2BOLG9f2FPJTiTAQhlkzYT2PxeknORz-ud',
        'referer':'https://www.google.com/',
        'x-devtools-emulate-network-conditions-client-id':'1A252FDF-B189-4B88-BFB0-0AEBB9856BD2',
        'user-agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 BIDUBrowser/7.6 Safari/537.36",
            'dnt':1
}
    tmStart = time.time();

    url = hosturl+"?"+req
    request = urllib2.Request(url,headers=headers)
    time.sleep(5)
    resp = urllib2.urlopen(request)
    html = resp.read()
    print >> sys.stderr,url
    print >> sys.stderr,"html query: \t",html

    if 1:
        simiurl = ""
        m = re.search("(tbs=simg:[^\"]+?)\">外观类似的图片", html)
        if m:
            simiurl = "https://www.google.com/search?"+(m.group(1)[:-1]).replace("amp;",'')
            print >> sys.stderr,"simiurl:",simiurl
            request = urllib2.Request(simiurl,headers=headers)
            time.sleep(5)
            resp = urllib2.urlopen(request)
            htmlsimi = resp.read()
            print >> sys.stderr,"html simi : \t",htmlsimi

            flag = False
            ei = ""
            ved = ""

            if 1:
                m = re.search("kEI:'([^\"]+?)'", htmlsimi)
                if m:
                    ei = m.group(1)
                    #print "ei",ei
                m = re.search("id=\"rg\" data-ved=\"([^\"]+?)\"", htmlsimi)
                if m:
                    ved = m.group(1)
                    #print "ved",ved

            resub = re.compile("&ved=.+?&")
            resub_1 = re.compile("%253.*")
            for ijn in xrange(1):
                para = {
                    "tbm":"isch",
                    "ijn": ijn,
                    "ei" : ei,
                    "start" : ijn * 100,
                    "ved": ved,
                    "vet" : "1"+ved+"."+ei+".i",
                    "sa" : "X",
                    "biw" : 1680,
                    "bih" : 570
                }


                if simiurl != "" and ved != "" and ei != "":
                    urltemp = (simiurl+"&").replace("&tbm=isch","")
                    url = resub.sub("&",urltemp) + urllib.urlencode( para )
                    print >> sys.stderr,"simiurl: ijn is %s .\t "%ijn,url

                    request = urllib2.Request(url,headers=headers)
                    resp = urllib2.urlopen(request)
                    htmltemp = resp.read()
                    print >> sys.stderr,"html num is %s .\t "%ijn,htmltemp

                    m = re.findall( "imgurl=http[^\"]+?&",htmltemp)
                    if m:
                        for ms in m:
                            print>> f,resub_1.sub("",ms[7:-1])
                time.sleep(3)

    tmEnd = time.time() - tmStart

if __name__ == '__main__':
    #for file in os.listdir(sys.argv[1]):
        #for i in xrange(0,2000,30):
    #url = sys.argv[1]
    '''
    url = "http://pic60.nipic.com/file/20150227/6842469_132105605000_2.jpg"
    url = "http://e.hiphotos.baidu.com/baike/c0%3Dbaike92%2C5%2C5%2C92%2C30/sign=e94bdded9925bc313f5009ca3fb6e6d4/42a98226cffc1e174e3c84124a90f603738de916.jpg"
    url = "http://bs.baidu.com/online-crowdtest/%2Feva20140606110150_a8882b96-a602-2b66-33f4-7842bcbb84b4_3"
    url = "http://img.chinawj.com.cn/picture/product/2013/13710/1009/94531445557.jpg"
    url = "http://www.nongshanghang.cn/uploads/allimg/c140709/1404W3Nb2E0-1G20.jpg"
    '''
    #print "[input url]", url
    with open(sys.argv[1],'r') as f:
        for i,line in enumerate(f):
            line =line.split('\t')
            try:
                getsimi(line[1],'urltxt/'+str(i).zfill(8)+'.txt')
            except:
                continue


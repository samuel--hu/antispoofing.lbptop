#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
# Thu Oct 02 12:29:00 CEST 2014
#
"""
"""
import unittest
from pkg_resources import resource_filename
import numpy
from  antispoofing.lbptop.spoof import *

class lbptopTest(unittest.TestCase):

  def test01_three_planes_uniform(self):

    volume = numpy.random.rand(3,10,10) #Just one output (3 "frames")

    nXY = 8; nXT = 8; nYT = 8;
    rX  = 1; rY  = 1; rT  = 1;
    cXY = False; cXT = False; cYT = False;
    lbptypeXY  = "uniform"; lbptypeXT  = "uniform"; lbptypeYT  = "uniform";
    elbptypeXY = "regular"; elbptypeXT = "regular"; elbptypeYT = "regular";

    histXY,histXT,histYT = lbptophist(volume,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,elbptypeXY,elbptypeXT,elbptypeYT)    

    assertXY = histXY.shape = (1,59)
    assertXT = histXT.shape = (1,59)
    assertYT = histYT.shape = (1,59)

    self.assertTrue(assertXY)
    self.assertTrue(assertXT)
    self.assertTrue(assertYT)
 


  def test02_three_planes_regular(self):

    volume = numpy.random.rand(4,10,10) #Just one output (4 "planes" in the time domain)

    nXY = 8; nXT = 8; nYT = 8;
    rX  = 1; rY  = 1; rT  = 1;
    cXY = False; cXT = False; cYT = False;
    lbptypeXY  = "regular"; lbptypeXT  = "regular"; lbptypeYT  = "regular";
    elbptypeXY = "regular"; elbptypeXT = "regular"; elbptypeYT = "regular";

    histXY,histXT,histYT = lbptophist(volume,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,elbptypeXY,elbptypeXT,elbptypeYT)    

    assertXY = histXY.shape = (2,256)
    assertXT = histXT.shape = (2,256)
    assertYT = histYT.shape = (2,256)

    self.assertTrue(assertXY)
    self.assertTrue(assertXT)
    self.assertTrue(assertYT)


  def test03_three_planes_regular_rt2(self):

    volume = numpy.random.rand(5,10,10) #Just one output (4 "planes" in the time domain)

    nXY = 8; nXT = 8; nYT = 8;
    rX  = 1; rY  = 1; rT  = 2;
    cXY = False; cXT = False; cYT = False;
    lbptypeXY  = "regular"; lbptypeXT  = "regular"; lbptypeYT  = "regular";
    elbptypeXY = "regular"; elbptypeXT = "regular"; elbptypeYT = "regular";

    histXY,histXT,histYT = lbptophist(volume,nXY,nXT,nYT,rX,rY,rT,cXY,cXT,cYT,lbptypeXY,lbptypeXT,lbptypeYT,elbptypeXY,elbptypeXT,elbptypeYT)    

    assertXY = histXY.shape = (1,256)
    assertXT = histXT.shape = (1,256)
    assertYT = histYT.shape = (1,256)

    self.assertTrue(assertXY)
    self.assertTrue(assertXT)
    self.assertTrue(assertYT)



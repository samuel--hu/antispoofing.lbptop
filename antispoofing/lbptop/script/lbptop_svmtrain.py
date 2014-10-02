#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiagofrepereira@gmail.com>
#Tue Jul 17 11:11:00 CEST 2012

"""This script can makes an SVM classification of data into two categories: real accesses and spoofing attacks for each LBP-TOP plane and it combinations. There is an option for normalizing between [-1, 1] and dimensionality reduction of the data prior to the SVM classification.
The probabilities obtained with the SVM are considered as scores for the data. Firstly, the EER threshold on the development set is calculated. The, according to this EER, the FAR, FRR and HTER for the test and development set are calculated. The script outputs a text file with the performance results.
The details about the procedure are described in the paper: "LBP-TOP based countermeasure against facial spoofing attacks" - de Freitas Pereira, Tiago and Anjos, Andre and De Martino, Jose Mario and Marcel, Sebastien; ACCV - LBP 2012
"""

import os, sys
import argparse
import bob
import numpy

from .. import spoof
from ..spoof import calclbptop

from antispoofing.utils.ml import *
from antispoofing.utils.db import *

from antispoofing.lbptop.helpers import *

def main():

  basedir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

  INPUT_DIR = os.path.join(basedir, 'lbp_features')
  OUTPUT_DIR = os.path.join(basedir, 'res')

  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)

  parser.add_argument('input_dir', metavar='DIR', type=str, default=INPUT_DIR, help='Base directory containing the scores to be loaded')

  parser.add_argument('output_dir', metavar='DIR', type=str, default=OUTPUT_DIR, help='Base directory that will be used to save the results.')

  parser.add_argument('-n', '--normalize', action='store_true', dest='normalize', default=False, help='If True, will do zero mean unit variance normalization on the data before creating the LDA machine')

  parser.add_argument('-r', '--pca_reduction', action='store_true', dest='pca_reduction', default=False, help='If set, PCA dimensionality reduction will be performed to the data before doing LDA')

  parser.add_argument('-e', '--energy', type=str, dest="energy", default='0.99', help='The energy which needs to be preserved after the dimensionality reduction if PCA is performed prior to LDA')

  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Increases this script verbosity')

  parser.add_argument('-s', '--score', dest='score', action='store_true', default=False, help='If set, the final classification scores of all the frames will be dumped in a file')

  #######
  # Database especific configuration
  #######
  Database.create_parser(parser, implements_any_of='video')

  args = parser.parse_args()
  inputDir      = args.input_dir
  outputDir     = args.output_dir
  verbose       = args.verbose
  score         = args.score

  if not os.path.exists(inputDir):
    parser.error("input directory does not exist")

  if not os.path.exists(outputDir): # if the output directory doesn't exist, create it
    bob.db.utils.makedirs_safe(outputDir)

  energy        = float(args.energy)
  normalize     = args.normalize
  pca_reduction = args.pca_reduction
  
  ##models = ['XY-plane','XT-Plane','YT-Plane','XT-YT-Plane','XY-XT-YT-plane']
  ##lines  = ['r','b','y','g^','c']

  #Normalization lowbound and highbound
  lowbound  = -1
  highbound = 1


  if(verbose):
    print "Loading input files..."

  ##########################
  # Loading the input files
  ##########################
  database = args.cls(args)
  trainReal, trainAttack = database.get_train_data()
  devReal, devAttack = database.get_devel_data()
  testReal, testAttack = database.get_test_data()

  # create the full datasets from the file data
  train_real_features = calclbptop.create_full_dataset(trainReal,inputDir); train_attack_features = calclbptop.create_full_dataset(trainAttack,inputDir); 
  dev_real_features   = calclbptop.create_full_dataset(devReal,inputDir);   dev_attack_features   = calclbptop.create_full_dataset(devAttack,inputDir); 
  test_real_features  = calclbptop.create_full_dataset(testReal,inputDir);  test_attack_features  = calclbptop.create_full_dataset(testAttack,inputDir); 

  ##########################
  # Training SVM
  ##########################

  if(verbose):
    print "Training SVM machine..."

  [svmMachine,pcaMachine,mins,maxs] = svmCountermeasure.train(train_real_features, train_attack_features, normalize=normalize, pca_reduction=pca_reduction,energy=energy)

  #Saving the machines
  if(pca_reduction):
    hdf5File_pca = bob.io.HDF5File(os.path.join(outputDir, 'pca_machine_'+ str(energy) + '.txt'),openmode_string='w')
    pcaMachine.save(hdf5File_pca)
    del hdf5File_pca

  svmMachine.save(os.path.join(outputDir, 'svm_machine.txt'))

  #Saving the normalization factors
  if(normalize):
    fileName = os.path.join(outputDir, 'svm_normalization.txt')
    svmCountermeasure.writeNormalizationData(fileName,lowbound,highbound,mins,maxs)


  if(pca_reduction):
    train_real_features   = pca.pcareduce(pcaMachine, train_real)
    train_attack_features = pca.pcareduce(pcaMachine, train_attack)
    dev_real_features     = pca.pcareduce(pcaMachine, dev_real)
    dev_attack_features   = pca.pcareduce(pcaMachine, dev_attack)
    test_real_features    = pca.pcareduce(pcaMachine, test_real)
    test_attack_features  = pca.pcareduce(pcaMachine, test_attack)

  train_real_out   = svmCountermeasure.svm_predict(svmMachine, train_real_features)
  train_attack_out = svmCountermeasure.svm_predict(svmMachine, train_attack_features)
  dev_real_out     = svmCountermeasure.svm_predict(svmMachine, dev_real_features)
  dev_attack_out   = svmCountermeasure.svm_predict(svmMachine, dev_attack_features)
  test_real_out    = svmCountermeasure.svm_predict(svmMachine, test_real_features)
  test_attack_out  = svmCountermeasure.svm_predict(svmMachine, test_attack_features)


  # calculation of the error rates
  thres              = bob.measure.eer_threshold(dev_attack_out.flatten(), dev_real_out.flatten())
  dev_far, dev_frr   = bob.measure.farfrr(dev_attack_out.flatten(), dev_real_out.flatten(), thres)
  test_far, test_frr = bob.measure.farfrr(test_attack_out.flatten(), test_real_out.flatten(), thres)

  # writing results to a file
  tbl = []
  tbl.append(" ")
  if args.pca_reduction:
    tbl.append("EER @devel - energy kept after PCA = %.2f" % (energy))
  tbl.append(" threshold: %.4f" % thres)
  tbl.append(" dev:  FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
      (100*dev_far, int(round(dev_far*len(dev_attack_features))), len(dev_attack_features), 
       100*dev_frr, int(round(dev_frr*len(dev_real_features))), len(dev_real_features),
       50*(dev_far+dev_frr)))
  tbl.append(" test: FAR %.2f%% (%d / %d) | FRR %.2f%% (%d / %d) | HTER %.2f%% " % \
      (100*test_far, int(round(test_far*len(test_attack_features))), len(test_attack_features),
       100*test_frr, int(round(test_frr*len(test_real_features))), len(test_real_features),
       50*(test_far+test_frr)))
  txt = ''.join([k+'\n' for k in tbl])
  print txt

 
if __name__ == '__main__':
  main()

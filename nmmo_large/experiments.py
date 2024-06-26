from pdb import set_trace as T
from configs import Law, Chaos
import os

#Oversimplified user specification
#for 1-3 collaborators
USER = 'your-username'
if USER == 'your-username':
   #Thousandth
   prefix = 'test'
   remote = False
   local  = not remote

   #下方三个变量是初始设置值，需要读取已有的model文件，来进行测试
   # test = True#local
   # best = True#local
   # load = True#local

   #进入训练的设定
   test = False#local
   best = False#local
   load = False#local


   #原有的初始值
   # sample = not test
   sample = False



   #原有的初始值
   # singles = True

   singles = False


   tournaments = False
   
   exps = {}
   # szs = [256]

   szs = [512]

   # szs = [128]

   #For full distributed runs
   #szs = (16, 32, 64, 128)
   names = 'law chaos'.split()
   confs = (Law, Chaos)

   def makeExp(name, conf, sz, test=False):
      NENT, NPOP = sz, sz//16
      ROOT = 'resource/exps/' + name + '/'
      try:
         os.mkdir(ROOT)
         os.mkdir(ROOT + 'model')
         os.mkdir(ROOT + 'train')
         os.mkdir(ROOT + 'test')
      except FileExistsError:
         pass
      MODELDIR = ROOT + 'model/'

      exp = conf(remote, 
            NENT=NENT, NPOP=NPOP,
            MODELDIR=MODELDIR,
            SAMPLE=sample,
            BEST=best,
            LOAD=load,
            TEST=test)
      exps[name] = exp
      print(name, ', NENT: ', NENT, ', NPOP: ', NPOP)

   def makeExps():
      #Training runs
      for label, conf in zip(names, confs):
         for sz in szs:
            name = prefix + label + str(sz)
            makeExp(name, conf, sz, test=test)
          
   #Sample config
   makeExps()
   makeExp('sample', Chaos, 128, test=False)

#Main file. Hooks into high level world/render updates
from pdb import set_trace as T
import argparse

import experiments
from forge.trinity import smith, Trinity, Pantheon, God, Sword
import time
import torch

def parseArgs():
   parser = argparse.ArgumentParser('Projekt Godsword')
   parser.add_argument('--nRealm', type=int, default='1', 
         help='Number of environments (1 per core)')
   parser.add_argument('--api', type=str, default='native', 
         help='API to use (native/vecenv)')
   parser.add_argument('--ray', type=str, default='default', 
         help='Ray mode (local/default/remote)')
   parser.add_argument('--render', action='store_true', default=False, 
         help='Render env')
   return parser.parse_args()

#Example runner using the (slower) vecenv api
#The actual vecenv spec was not designed for
#multiagent, so this is a best-effort facsimile
class GymExample:
   def __init__(self, config, args):
      self.env = smith.VecEnv(config, args, self.step)
      #The environment is persistent. Reset only to start it.
      self.envsObs = self.env.reset()
      #the ANN used internally by Trinity
      from forge.trinity import ANN
      self.ann = ANN(config)
      self.stepTotalTime = 0
      self.runTime = 0

   #Runs a single step of each environment
   #With slow comms at each step
   def step(self):
      runStartTime = time.time()
      actions = []
      for obs in self.envsObs: #Environment
         atns = []
         for ob in obs: #Agent
            ent, stim = ob
            action, arguments, atnArgs, val = self.ann(ent, stim)
            atns.append((ent.entID, action, arguments, float(val)))
         actions.append(atns)
      stepStartTime = time.time()
      self.envsObs, rews, dones,infos = self.env.step(actions)
      stepEndTime = time.time()
      runEndTime = time.time()
      self.stepTotalTime += (stepEndTime-stepStartTime)
      self.runTime += (runEndTime - runStartTime)
      print("stepTotalTime:",self.stepTotalTime,"runTime:",self.runTime)

   def run(self):


      index = 0

      maxlen = 0

      count_index = 0
      while True:

         if index == 100:
            count_index += index
            index = 0
            continue

         self.step()
         if maxlen <= len(self.envsObs[0]):
            maxlen = len(self.envsObs[0])
         index += 1

class NativeExample:
   def __init__(self, config, args):
      trinity = Trinity(Pantheon, God, Sword)
      self.env = smith.Native(config, args, trinity)

   def run(self):
      while True:
         self.env.run()

if __name__ == '__main__':
   args = parseArgs()
   assert args.api in ('native', 'vecenv')
   # config = experiments.exps['testchaos128']
   config = experiments.exps['testlaw128']
   # config = experiments.exps['testlaw256']

   # config = experiments.exps['testlaw512']

   # config = experiments.exps['testlaw16']


   # config = experiments.exps['sample']

   if args.api == 'native':
      example = NativeExample(config, args)
   elif args.api == 'vecenv':
      example = GymExample(config, args)

   #Rendering by necessity snags control flow
   #This will automatically set local mode with 1 core
   if args.render:
      example.env.render()
   
   example.run()

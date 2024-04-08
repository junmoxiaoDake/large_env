import ray
import pickle
import numpy as np
from pdb import set_trace as T
import numpy as np

from forge import trinity as Trinity
from forge.blade import entity, core
from itertools import chain
from copy import deepcopy
import time
from multiprocessing import Process

class ActionArgs:
   def __init__(self, action, args):
      self.action = action
      self.args = args

class Realm:
   def __init__(self, config, args, idx):
      #Random samples
      if config.SAMPLE:
         config = deepcopy(config)
         nent = np.random.randint(0, config.NENT)
         config.NENT = config.NPOP * (1 + nent // config.NPOP)
      self.world, self.desciples = core.Env(config, idx), {}
      self.config, self.args, self.tick = config, args, 0
      self.npop = config.NPOP

      self.env = self.world.env
      self.values = None

   def clientData(self):
      if self.values is None and hasattr(self, 'sword'):
         self.values = self.sword.anns[0].visVals()

      ret = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) for k, v in self.desciples.items()),
            'values': self.values
            }
      return pickle.dumps(ret)

   def spawn(self):
      if len(self.desciples) >= self.config.NENT:
         return

      entID, color = self.god.spawn()
      ent = entity.Player(entID, color, self.config)
      self.desciples[ent.entID] = ent

      r, c = ent.pos
      self.world.env.tiles[r, c].addEnt(entID, ent)
      self.world.env.tiles[r, c].counts[ent.colorInd] += 1

   def cullDead(self, dead):
      for entID in dead:
         ent = self.desciples[entID]
         r, c = ent.pos
         self.world.env.tiles[r, c].delEnt(entID)
         self.god.cull(ent.annID)
         del self.desciples[entID]

   def stepWorld(self):
      ents = list(chain(self.desciples.values()))
      self.world.step(ents, [])

   def stepEnv(self):
      self.world.env.step()
      self.env = self.world.env.np()

   def stepEnt(self, ent, action, arguments):
      move, attack         = action
      moveArgs, attackArgs = arguments

      ent.move   = ActionArgs(move, moveArgs)
      ent.attack = ActionArgs(attack, attackArgs[0])

   def getStim(self, ent):
      return self.world.env.stim(ent.pos, self.config.STIM)

@ray.remote
class NativeRealm(Realm):
   def __init__(self, trinity, config, args, idx):
      super().__init__(config, args, idx)
      self.god = trinity.god(config, args)
      self.sword = trinity.sword(config, args)
      self.sword.anns[0].world = self.world
 
   def stepEnts(self):
      dead = []
      for ent in self.desciples.values():
         ent.step(self.world)

         if self.postmortem(ent, dead):
            continue

         stim = self.getStim(ent)
         action, arguments, val = self.sword.decide(ent, stim)
         ent.act(self.world, action, arguments, val)

         self.stepEnt(ent, action, arguments)

      self.cullDead(dead)

   def postmortem(self, ent, dead):
      entID = ent.entID
      if not ent.alive or ent.kill:
         dead.append(entID)
         if not self.config.TEST:
            self.sword.collectRollout(entID, ent)
         return True
      return False

   def step(self):
      self.spawn()
      self.stepEnv()
      self.stepEnts()
      self.stepWorld()

   def run(self, swordUpdate=None):
      self.recvSwordUpdate(swordUpdate)

      updates = None
      while updates is None:
         self.step()
         updates, logs = self.sword.sendUpdate()
      return updates, logs

   def recvSwordUpdate(self, update):
      if update is None:
         return
      self.sword.recvUpdate(update)

   def recvGodUpdate(self, update):
      self.god.recv(update)

# @ray.remote(num_gpus=1)
@ray.remote
class VecEnvRealm(Realm):
   #Use the default God behind the scenes for spawning
   def __init__(self, config, args, idx):
      super().__init__(config, args, idx)
      self.god = Trinity.God(config, args)
      self.loadTime = 0

      self.stepEntsTime = 0
      self.stepWorldTime = 0
      self.spawnTime = 0
      self.EnvTime = 0
      self.returnTime = 0

   def stepEnts(self, decisions):
      dead = []
      for tup in decisions:
         entID, action, arguments, val = tup
         ent = self.desciples[entID]
         ent.step(self.world)

         if self.postmortem(ent, dead):
            continue

         ent.act(self.world, action, arguments, val)
         self.stepEnt(ent, action, arguments)
      self.cullDead(dead)

   def postmortem(self, ent, dead):
      entID = ent.entID
      if not ent.alive or ent.kill:
         dead.append(entID)
         return True
      return False

   def step(self, decisions):
      loadStartTime = time.time()
      decisions = pickle.loads(decisions)
      loadEndTime = time.time()

      self.loadTime += (loadEndTime - loadStartTime)

      entsStartTime = time.time()
      self.stepEnts(decisions)
      entsEndTime = time.time()

      self.stepEntsTime += (entsEndTime - entsStartTime)

      worldStartTime = time.time()
      self.stepWorld()
      worldEndTime = time.time()

      self.stepWorldTime += (worldEndTime - worldStartTime)

      spawnStartTime = time.time()
      self.spawn()
      spawnEndTime = time.time()

      self.spawnTime += (spawnEndTime - spawnStartTime)

      envStartTime = time.time()
      self.stepEnv()
      envEndTime = time.time()
      self.EnvTime += (envEndTime - envStartTime)

      returnStartTime = time.time()
      stims, rews, dones = [], [], []
      for entID, ent in self.desciples.items():
         stim = self.getStim(ent)
         stims.append((ent, self.getStim(ent)))
         rews.append(1)
      returnEndTime = time.time()
      self.returnTime += (returnEndTime - returnStartTime)
      return pickle.dumps((stims, rews, None, None, self.loadTime, self.stepEntsTime, self.stepWorldTime, self.spawnTime, self.EnvTime, self.returnTime))

   def reset(self):
      print("我进来了？")
      for i in range(self.config.NENT):  # 让智能体初始就达到config设定的最大个数
         self.spawn()
      self.stepEnv()
      return [(e, self.getStim(e)) for e in self.desciples.values()]



class LocalVecRealm(Realm):
   #Use the default God behind the scenes for spawning
   def __init__(self, config, args, idx):
      super().__init__(config, args, idx)
      self.god = Trinity.God(config, args)

      self.loadTime = 0

      self.stepEntsTime = 0
      self.stepWorldTime = 0
      self.spawnTime = 0
      self.EnvTime = 0
      self.returnTime = 0

   def process_data(self, tup, deads):
      # 处理数据的逻辑
      entID, action, arguments, val = tup
      ent = self.desciples[entID]
      ent.step(self.world)
      if self.postmortem(ent, deads):
         return
      ent.act(self.world, action, arguments, val)
      self.stepEnt(ent, action, arguments)



   def stepEnts(self, decisions):
      before_time_s = time.time()
      dead = []
      for tup in decisions:
         entID, action, arguments, val = tup
         ent = self.desciples[entID]
         ent.step(self.world)

         if self.postmortem(ent, dead):
            continue

         ent.act(self.world, action, arguments, val)
         self.stepEnt(ent, action, arguments)

      before_time_e = time.time()
      before_time = before_time_e - before_time_s

      # num_processes = 4  # 你可以根据需要调整进程数量
      # after_time_s = time.time()
      # processes = []
      # deads = []
      # for tup in decisions:
      #    process = Process(target=self.process_data, args=(tup,deads))
      #    processes.append(process)
      #    process.start()
      #
      # # 等待所有进程完成
      # for process in processes:
      #    process.join()
      #
      # after_time_e = time.time()
      # after_time = after_time_e - after_time_s
      #
      # print("加速的倍数：",before_time/after_time,"before:",before_time,"after:",after_time)

      self.cullDead(dead)

   def postmortem(self, ent, dead):
      entID = ent.entID
      if not ent.alive or ent.kill:
         dead.append(entID)
         return True
      return False

   def step(self, decisions):
      loadStartTime = time.time()
      decisions = pickle.loads(decisions)
      loadEndTime = time.time()

      self.loadTime += (loadEndTime - loadStartTime)


      entsStartTime = time.time()
      self.stepEnts(decisions)
      entsEndTime = time.time()

      self.stepEntsTime += (entsEndTime - entsStartTime)

      worldStartTime = time.time()
      self.stepWorld()
      worldEndTime = time.time()

      self.stepWorldTime += (worldEndTime - worldStartTime)

      spawnStartTime = time.time()
      self.spawn()
      spawnEndTime = time.time()

      self.spawnTime += (spawnEndTime - spawnStartTime)

      envStartTime = time.time()
      self.stepEnv()
      envEndTime = time.time()
      self.EnvTime += (envEndTime - envStartTime)

      returnStartTime = time.time()
      stims, rews, dones = [], [], []
      for entID, ent in self.desciples.items():
         stim = self.getStim(ent)
         stims.append((ent, self.getStim(ent)))
         rews.append(1)
      returnEndTime = time.time()
      self.returnTime += (returnEndTime - returnStartTime)

      print("------>loadTime:",self.loadTime,"stepEntsTime:", self.stepEntsTime, "stepWorldTime:", self.stepWorldTime,"spawnTime:",self.spawnTime,"EnvTime:",self.EnvTime,"returnTime",self.returnTime)

      return pickle.dumps((stims, rews, None, None,self.loadTime))

   def reset(self):
      for i in range(self.config.NENT):#让智能体初始就达到config设定的最大个数
         self.spawn()
      self.stepEnv()
      return [(e, self.getStim(e)) for e in self.desciples.values()]

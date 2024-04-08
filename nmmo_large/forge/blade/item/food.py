from forge.blade.systems import Skill
from forge.blade.item import Item

class Food(Item.Item):
   createSkill = Skill.Cooking
   useSkill = Skill.Constitution
   heal = None

class Ration(Food):
   useLevel = 1
   exp = 0
   heal = 1

class Shrimp(Food):
   createLevel = 1
   useLevel = 1
   exp = 10
   heal = 2

class Sardine(Food):
   createLevel = 5
   useLevel = 5
   exp = 20
   heal = 3

class Herring(Food):
   createLevel = 10
   useLevel = 10
   exp = 30
   heal = 5

class Chicken(Food):
   createLevel = 1
   useLevel = 1
   exp = 10
   heal = 3

class Goblin(Food):
   createLevel = 1
   useLevel = 5
   exp = 10
   heal = 5

class Drug(Food):
   createLevel = 35
   useLevel = 15
   exp = 100
   heal = 50

class Medical_Kit(Food):
   createLevel = 50
   useLevel = 25
   exp = 150
   heal = 100

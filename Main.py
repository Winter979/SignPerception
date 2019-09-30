'''
File: Main.py
File Created: Monday, 30th September 2019 7:49:43 pm
Author: Jonathon Winter
-----
Last Modified: Monday, 30th September 2019 11:29:40 pm
Modified By: Jonathon Winter
-----
Purpose: 
'''

import sys
import argparse
import glob

from Settings import Settings

# import Task1 as t1
# import Task2 as t2
import Task as t

def Setup_Args():
   parser = argparse.ArgumentParser(description="Analyze Packet Dump for anomalies")
   parser.add_argument("-t","--task", dest="task", help="The task that is to be executed")
   parser.add_argument("-f","--files", dest="files", help="The files that will be tested against")
   parser.add_argument("-s","--show",dest="show", action="store_true", help="Display the images")
   # parser.add_argument("-v","--verbose",dest="verbose", action="store_true", help="Enable verbose mode")
   parser.add_argument("--test",dest="test", action="store_true", help="Runs with the test images")
   # parser.add_argument("-m","--manual", dest="manual", help="Manually select a single image")
   return parser.parse_args()

def Get_Files(task_no, file_no):

   if task_no == 1:
      files = glob.glob("{}/*".format(Settings.T1_Images))
   elif task_no == 2:
      files = glob.glob("{}/*".format(Settings.T2_Images))
   else:
      raise ValueError("There are only 2 tasks available")

   if file_no != None:
      if "-" in file_no:
         if file_no.startswith("-"):
            file_no = int(file_no[1:])-1
            files = files[:file_no]
         elif file_no.endswith("-"):
            file_no = int(file_no[:-1])-1
            files = files[file_no:]
         else:
            split = file_no.split("-")
            files = files[int(split[0])-1:int(split[1])]
      else:
         file_no = int(file_no)-1
         files = [files[file_no]]

   return files

if __name__ == "__main__":
   args = Setup_Args()

   Settings.task = int(args.task)
   Settings.test = int(args.test)

   if Settings.test:
      files = Get_Files(Settings.task,args.files)

   Settings.show = args.show

   t.Main(files)
   
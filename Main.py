import sys
import argparse
import glob

from Settings import Settings as s

import Task1 as t1
import Task2 as t2

def Setup_Args():
   parser = argparse.ArgumentParser(description="Analyze Packet Dump for anomalies")
   parser.add_argument("-t","--task", dest="task", help="The task that is to be executed")
   parser.add_argument("-f","---files", dest="files", help="The files that will be tested against")
   parser.add_argument("-v","--verbose",dest="verbose", action="store_true", help="Enable verbose mode")
   parser.add_argument("-s","--show",dest="show", action="store_true", help="Display the images")
   parser.add_argument("-m","--manual", dest="manual", help="Manually select a single image")
   parser.add_argument("--train",dest="train",action="store_true",help="Manually enter the results to train it")
   return parser.parse_args()

def Get_Files(task_no, file_no):

   if task_no == 1:
      files = glob.glob("{}/*".format(s.T1_Images))
   elif task_no == 2:
      files = glob.glob("{}/*".format(s.T2_Images))
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

   # try:

   task_no = int(args.task)

   files = Get_Files(task_no,args.files)

   s.show = args.show
   s.train = args.train

   if task_no == 1:
      t1.Main(files)
   elif task_no == 2:
      t2.Main(files)
   # except Exception as e:
   #    print(e)
   
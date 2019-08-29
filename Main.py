import sys
import argparse
import glob

import Task1 as t1
import Task2 as t2

def Setup_Args():
   parser = argparse.ArgumentParser(description="Analyze Packet Dump for anomalies")
   parser.add_argument("-t","--task", dest="task", help="The task that is to be executed")
   parser.add_argument("-f","---files", dest="files", help="The files that will be tested against")
   parser.add_argument("-v","--verbose",dest="verbose", action="store_true", help="Enable verbose mode")
   parser.add_argument("-m","--manual", dest="manual", help="Manually select a single image")
   return parser.parse_args()

def Get_Files(task_no, file_no):

   if task_no == 1:
      files = glob.glob("BuildingSignage/*")
   elif task_no == 2:
      files = glob.glob("DirectionalSignage/*")
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

   if task_no == 1:
      t1.main(files)
   elif task_no == 2:
      t2.main(files)
   # except Exception as e:
   #    print(e)
   
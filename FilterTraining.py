if __name__ == "__main__":
   data = []
   with open("full.txt") as f:
      lines = f.read().splitlines()
      for line in lines:
         res = int(line[0])
         ratios = [float(ii) for ii in line[2:].split(",")]
         data.append([res,ratios])

      
      temp = [ii[0] for ii in data]
      mn = len(temp)
      for ii in range(10):
         cnt = temp.count(ii)
         mn = cnt if cnt < mn else mn

      print(mn)

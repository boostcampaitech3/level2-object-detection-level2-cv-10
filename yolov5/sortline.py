file_name = "submissions/Yolo_final"

with open(file_name+'.txt', 'r') as f:
    tmp = f.readlines()

result = sorted(tmp, key=lambda x: int(x.split(',')[0]))  

final_result = []

for line in result:
    line = line.split(',')
    prefix = line[0]
    postfix = line[2]
    tmp = (line[1]).split('|')[:-1]
    tmp = sorted(tmp, key = lambda x: x[0])
    tmp = prefix + ',' + ''.join(tmp) + ',' +postfix 
    final_result.append(tmp)
final_result.insert(0, "PredictionString,image_id\n")
   


with open(file_name+'.csv', 'w') as f:
     f.writelines(final_result)

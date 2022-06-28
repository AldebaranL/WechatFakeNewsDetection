import json
c_list=[[[1,2],[3,4],[5,6]],[[7,8],[9],[10]]]
c_list = json.dumps(c_list)
'''将c_list存入文件
'''
a = open("feature.txt", "w",encoding='UTF-8')
a.write(c_list)
a.close()

'''读取data_source_list文件
'''
b = open("feature.txt", "r",encoding='UTF-8')
out = b.read()
out =  json.loads(out)
print(out)
print(isinstance(out,list))
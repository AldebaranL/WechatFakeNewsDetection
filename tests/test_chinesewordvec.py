f = open("../datasets/sgns.sogounews.bigram-char","r",encoding='UTF-8')   #设置文件对象
t=f.readline().split()
n,dimension=int(t[0]),int(t[1])
print(dimension)
chinesewordvec=f.readlines()
chinesewordvec=[i.split() for i in chinesewordvec]
data=[]
wordtoindex={}
indextoword={}
for i in range(n):
    data.append(list(map(float,chinesewordvec[i][len(chinesewordvec[i])-dimension:])))
    wordtoindex[chinesewordvec[i][0]]=i
    indextoword[i]=chinesewordvec[i][0]
f.close()
print("finish reading cwv")
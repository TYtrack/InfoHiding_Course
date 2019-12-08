init=0.633
list_hun=[]
len_cipher=10
for i in range(len_cipher):
    init=round(1-init*init*2,4)
    list_hun.append(init)

list_pai=sorted(list_hun)
list_W=[ list_hun.index(i)for i in list_pai]
'''
    list_W:list_init每个元素在list_pai的索引位置
    list_pai:排序的混沌序列
    list_init:初始混沌序列
'''

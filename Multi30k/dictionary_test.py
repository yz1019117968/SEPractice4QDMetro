# 字典初始化
dict_test = {'b':2, 'c':3}
dict_digts = {1:'123', 2:'234'}

# 字典更新
dict_test.update({'a':1})
dict_test['d'] = 4
# 给定key修改value
dict_test['b'] = 22
print("after modification: ", dict_test)

# 删除元素
del dict_test['d']
print("after del: ", dict_test)

# 字典遍历
for key in dict_test:
    print(f"key: {key}; value: {dict_test[key]}")

# 返回字典长度
print(len(dict_test))

#返回执行key的value，若没有则返回default值
print("c-value: ", dict_test.get('c'))
print("d-value: ", dict_test.get('d'))

# 参考： https://www.runoob.com/python/python-dictionary.html
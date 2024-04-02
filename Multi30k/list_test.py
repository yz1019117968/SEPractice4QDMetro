# 列表的数据项不需要具有相同的类型
# 创建一个列表，只要把逗号分隔的不同的数据项使用方括号括起来即可。如下所示：
list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]

# 列表选取元素
print("list1[0]: ", list1[0])
print("list2[1:5]: ", list2[1:5])

# 更新元素
list1[0] = 'math'
print(list1)

# 添加元素
list = []          ## 空列表
list.append('Google')   ## 使用 append() 添加元素
list.append('Runoob')
print(list)

# 删除元素
del list[1]
print("after del: ", list)

# 拼接列表
print("list: ", list)
print("list1: ", list1)
new_list = list + list1
print("list + list1: ", new_list)

#返回列表长度，最大值，最小值
print("length list2: ", len(list2))
print("list2 max value: ", max(list2))
print("list2 min value: ", min(list2))

#统计某元素出现次数
print("freq of 5: ", list2.count(5))

# 从列表中找出某个值第一个匹配项的索引位置
print("index of 5: ", list2.index(5))

# 逆序列表
list2.reverse()
print("reverse list2: ", list2)

# 对数组排序
list_sort = [1,4,6,3,2,6,9,10]
list_sort.sort(reverse=True)
print(list_sort)

# 参考： https://www.runoob.com/python/python-lists.html
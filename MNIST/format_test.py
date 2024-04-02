# 指定/不指定顺序
print("{} {}".format("hello", "world"))
print("{1} {0}".format("world", "hello"))

print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))

# 通过字典设置参数
site = {"name": "菜鸟教程", "url": "www.runoob.com"}
print("网站名：{name}, 地址 {url}".format(**site))

# 通过列表索引设置参数
my_list = ['菜鸟教程', 'www.runoob.com']
my_list1 = ['test', 'one']
print("网站名：{0[0]}, 地址 {1[0]}".format(my_list, my_list1))

# 通过字典设置参数
site = {"name": "菜鸟教程", "url": "www.runoob.com"}
print("网站名：{name}, 地址 {url}".format(**site))

# 保留小数点后2/0位，四舍五入
print("{:.2f}, {:.0f}".format(3.1415926, 3.1415))

# 带符号保留小数点后2位，四舍五入
print("{:+.2f}, {:-.2f}".format(3.1415926, -3.1415926))

# 带百分号格式
print("{:+.2%}, {:.2%}".format(0.31415926, 0.2324))



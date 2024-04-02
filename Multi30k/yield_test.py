
# yield
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:", res)
g = foo()
print(next(g))
print("*" * 20)
print(next(g))

# 首先，如果你还没有对yield有个初步分认识，那么你先把yield看做“return”，这个是直观的，它首先是个return，普通的return是什么意思，就是在程序中返回某个值，返回之后程序就不再往下运行了。
# 看做return之后再把它看做一个是生成器（generator）的一部分（带yield的函数才是真正的迭代器），生成器是一种迭代器
# 我们先把yield看做return,然后直接看下面的程序，你就会明白yield的全部意思了：

def foo(num):
    print("starting...")
    while num<10:
        num=num+1
        yield num
# 节省内存
for n in foo(0):
    print(n)
for n in range(10):
    print(n)
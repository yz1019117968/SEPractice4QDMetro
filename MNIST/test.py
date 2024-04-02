# 面向对象最重要的概念就是类（Class）和实例（Instance），必须牢记类是抽象的模板，比如Student类，而实例是根据类创建出来的一个个具体的“对象”，每个对象都拥有相同的方法，但各自的数据可能不同。
# 通常，如果没有合适的继承类，就使用object类，这是所有类最终都会继承的类。
class Student(object):

    def __init__(self, name, score):
        self.name = name #属性
        self.score = score

# 注意到__init__方法的第一个参数永远是self，表示创建的实例本身，因此，在__init__方法内部，就可以把各种属性绑定到self，因为self就指向创建的实例本身。
#
# 有了__init__方法，在创建实例的时候，就不能传入空的参数了，必须传入与__init__方法匹配的参数，但self不需要传，Python解释器自己会把实例变量传进去：

bart = Student('Bart Simpson', 59)
print(bart.name)

# 和普通的函数相比，在类中定义的函数只有一点不同，就是第一个参数永远是实例变量self，并且，调用时，不用传递该参数。除此之外，类的方法和普通函数没有什么区别，所以，你仍然可以用默认参数、可变参数、关键字参数和命名关键字参数。

class Student1(object):

    def __init__(self, name, score):
        self.name = name #属性
        self.score = score

    def print_score(self):
        print(f"name: {self.name}, score: {self.score}")

    def get_grade(self):
        if self.score >= 90:
            return 'A'
        elif self.score >= 60:
            return 'B'
        else:
            return 'C'

def sum_n(n):
    sum = 0
    for i in range(n):
       sum += i
    print("sum: ", sum)

def print_n(n):
    lst = []
    for i in range(n):
        lst.append(i)
    print(lst)
# 要调用一个方法，只需要在实例变量上直接调用，除了self不用传递，其他参数正常传入：
stu = Student1("a", 30)
stu.print_score()
sum_n(5)
print_n(5)


import torch

pred = torch.tensor([[1],[2],[3],[4],[5]])
label = torch.tensor([[1],[3],[4],[5],[6]])
print(pred.eq(label).sum())
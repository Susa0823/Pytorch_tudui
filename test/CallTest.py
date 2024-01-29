class Person:
    def __call__(self, name):
        print("__cal__"+"Hello" + name)

    def hello(self, name):
        print("hello" + name)

Person = Person()

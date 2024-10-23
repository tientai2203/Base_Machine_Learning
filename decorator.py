def repeat(num_times):
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                func(*args, **kwargs)
        return wrapper
    return decorator_repeat

@repeat(num_times=3)
def greet(name):
    print(f"Hello {name}")

greet("Alice")


# functools : functools.wraps là một decorator được sử dụng để bảo tồn thông tin của hàm gốc khi bạn tạo một hàm wrapper
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Calling decorated function")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
print(say_hello.__name__)  # Output: say_hello

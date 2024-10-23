def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

# Sử dụng generator
counter = count_up_to(5)
print(counter)
print(list(counter))
for number in counter:
    print(number)
print(counter)
print(list(counter))

###
def simple_gen():
    value = 1
    while value < 5:
        received = yield value
        if received:
            print(f"Received: {received}")
        value += 1

gen = simple_gen()
print(next(gen))  # 1
print(gen.send('Hello'))  # Received: Hello \n 2
print(next(gen))  # 3
print(next(gen))  # 4
print(next(gen)) 
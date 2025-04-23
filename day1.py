# function to add two numbers
def add(x, y):
    return x + y;

# function to subtract two numbers
def subtract(x, y):
    return x - y;

# function to multiply two numbers
def multiply(x, y):
    return x * y;

# function to divide two numbers - return an error string if division by zero is attempted
def divide(x, y):
    if y == 0:
        return "Error! Division by zero."
    else:
        return x / y;


def calculator():
    print("Select operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    while True:
        # take input from the user
        choice = input("Enter choice (1/2/3/4): ")

        # check if choice is one of the four options
        if choice in ['1', '2', '3', '4']:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))

            if choice == '1':
                print(f"{num1} + {num2} = {add(num1, num2)}")
            elif choice == '2':
                print(f"{num1} - {num2} = {subtract(num1, num2)}")
            elif choice == '3':
                print(f"{num1} * {num2} = {multiply(num1, num2)}")
            elif choice == '4':
                print(f"{num1} / {num2} = {divide(num1, num2)}")
        else:
            print("Invalid Input")

        # check if the user wants to perform another calculation
        next_calculation = input("Do you want to perform another calculation? (yes/no): ")
        if next_calculation.lower() == 'no':
            break

    print("Exiting the calculator. Goodbye!")

# Run the calculator function
calculator()
while True:
    try:
        x = int(input('Enter a number: '))
        break
    except:
        print("That's not a valid number")
    finally:
        print('Attempted input')

# Below are examples of how to handle some specific exception

# try:
#     # some code
# except ValueError:
#     # some code

# try:
#     # some code
# except (ValueError, KeyboardInterrupt):
#     # some code

# try:
#     # some code
# except ValueError:
#     # some code
# except KeyboardInterrupt:
#     # some code
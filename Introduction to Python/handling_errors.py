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

# try:
#     # some code
# except ZeroDivisionError as e:
#    # some code
#    print("ZeroDivisionError occurred: {}".format(e))

# try:
#     # some code
# except Exception as e:
#    # some code
#    print("Exception occurred: {}".format(e))
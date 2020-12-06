f = open('message.py', 'r')
file_data = f.read() #read a file and return string
f.close() #free up any existing resources taken up by the file

print(file_data)

#If the file doesn't exist, 'w' mode creates it
f = open('another_file.txt', 'w')
f.write('Hello World!')
f.close()
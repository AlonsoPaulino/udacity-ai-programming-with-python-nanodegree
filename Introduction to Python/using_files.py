f = open('message.py', 'r')
file_data = f.read() #read a file and return string
f.close() #free up any existing resources taken up by the file

print(file_data)

#If the file doesn't exist, 'w' mode creates it
f = open('another_file.txt', 'w')
f.write('Hello World!')
f.close()

#with keyboard allows you to open a file, perform operations and automatically close it
with open('another_file.txt', 'r') as f:
    file_data = f.read()

print(file_data)

#Each time we called read on the file with an integer argument, it read up to that number 
#of characters, outputted them, and kept the 'window' at that position for the next call to read
with open("another_file.txt") as song:
    print(song.read(2))
    print(song.read(8))
    print(song.read())


#Iterating through each file line. Deleting the end line using strip().
camelot_lines = []
with open("camelot.txt") as f:
    for line in f:
        camelot_lines.append(line.strip())

print(camelot_lines)
names = input("Enter a list of names: ").split(',') # get and process input for a list of names
assignments =  input("Enter the list of missing assignments: ").split(',') # get and process input for a list of the number of assignments
grades = input("Enter the list of grades: ").split(',') # get and process input for a list of grades

# message string to be used for each student
# HINT: use .format() with this string in your for loop
message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. You're current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

rng = range(0, len(names))
for i in rng:
    print(message.format(names[i], assignments[i], grades[i], int(grades[i]) + 2 * int(assignments[i])))

# write a for loop that iterates through each set of names, assignments, and grades to print each student's message


# Alternative solution using zip (more elegant)
for name, assignment, grade in zip(names, assignments, grades):
    print(message.format(name, assignment, grade, int(grade) + int(assignment) * 2))
import sys


def stairs(n, position, operation):
    output = [-1 for _ in range(n)]

    # WRITE YOUR CODE HERE
    # (and also feel free to write outside this function,
    # for example, if you want to write new functions).

    return output


# Set SUBMIT_TO_SZKOPUL=True when submitting
# your solution to the Szkopul webserver.
# Set SUBMIT_TO_SZKOPUL=False in order
# to test your code by reading the input from
# a test file ("input.txt").
SUBMIT_TO_SZKOPUL = True

if SUBMIT_TO_SZKOPUL:
    reader = sys.stdin
else:
    reader = open("input.txt", "r")

# Reads the input
astr = reader.readline().split()
n = int(astr[0])
positions = [int(val) for val in reader.readline().split()]
operations = [int(val) for val in reader.readline().split()]

if not SUBMIT_TO_SZKOPUL:
    reader.close()
    
# Calls your function
output = stairs(n, positions, operations)

# Writes the output
for value in output:
    print(value, end=" \n")  # Adds a trailing whitespace as per the format rules

import numpy as np
fname = open('D-Random-Multi.txt')
arr = fname.read()
a = []
i = -1
y = True
while i < len(arr):
    i = i+1
    buffer = ''
    if (arr[i] >='0' and arr[i] <='9') or arr[i] == '.':
        x = True
    while x:
        buffer =  buffer + arr[i]
        i = i + 1
        if i == len(arr) or arr[i] == ' ' or arr[i] == '\n':
            x = False
            a.append(buffer)
print (len(a))
array = np.array([[0.0,0.0,0.0]] * 25001)
for i in range(25001):
    array[i][0] = float(a[3*i + 0])
    array[i][1] = float(a[3*i + 1])
    array[i][2] = float(a[3*i + 2])
print (array)

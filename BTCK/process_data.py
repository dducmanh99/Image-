#
# with open ('ref.txt','r') as f:
#     for line in f:
#         # print (line)
#         a = line.split()
        
#         with open('ref2.txt','a') as g:
#             g.write(a[0]+' '+a[1]+' '+' '+str(float(a[0])*2.5)+' '+str(float(a[1])*2.5)+'\n')

#
A = []
with open('ref2.txt','r') as f:
    for line in f:
        a = line.split()
        A.append(a)
        

print(float(A[1]))


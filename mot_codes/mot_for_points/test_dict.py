
a = {}
a['1'] = [1,2]
a['2'] = [3,4]
a['3'] = [5]
a['4'] = [6]
b = a.copy()
for id in a.keys():
    if len(a[id]) ==1:
        b.pop(id)

print(a)
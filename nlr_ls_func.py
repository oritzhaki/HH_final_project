import random
import numpy as np

def l2(vec1, vec2):
    loss = 0
    for i in range(0,len(vec1)):
        ans = 0
        ans += (vec1[i] - vec2[i]) ** 2
        loss += ans
        # print(f"vec1[i]: {vec1[i]} vec2[i]: {vec2[i]} ans: {ans}")
    return loss    

def l1(vec1, vec2):
    loss = 0
    for i in range(0,len(vec1)):
        ans = 0
        ans += np.abs((vec1[i] - vec2[i]))
        loss += ans
        # print(f"vec1[i]: {vec1[i]} vec2[i]: {vec2[i]} ans: {ans}")
    return loss

loss_dict = {}

for i in range(0, 100):
    # Generate a list of 10 random numbers between 1 and 10
    vec1 = [random.randint(1, 10) for i in range(9)]
    vec2 = [random.randint(1, 10) for i in range(9)]
    l1_loss = l1(vec1,vec2)
    l2_loss = l2(vec1,vec2)
    loss_dict[i] = {'Vec1': vec1, 'Vec2': vec2, 'L1': l1_loss, 'L2': l2_loss}

# sort the dictionary by L1 loss and L2 loss values
sorted_dict_L1 = sorted(loss_dict.items(), key=lambda x: x[1]['L1'])
sorted_dict_L2 = sorted(loss_dict.items(), key=lambda x: x[1]['L2'])

# print the sorted dictionaries side by side
print('{:<25} {:<25} {:<15} {:<15} {:<15} {:<15}'.format('Vec1', 'Vec2', 'L1 Loss', 'Index', 'L2 Loss', 'Index'))
for (k1, v1), (k2, v2) in zip(sorted_dict_L1, sorted_dict_L2):
    vec1_str = str(v1['Vec1']).replace(",", "")[1:-1]
    vec2_str = str(v1['Vec2']).replace(",", "")[1:-1]
    print('{:<25} {:<25} {:<15} {:<15} {:<15} {:<15}'.format(vec1_str, vec2_str, v1['L1'], k1, v2['L2'], k2))

# check if the order of losses is preserved
print("Is the order of L1 losses preserved? ", [i[0] for i in sorted_dict_L1] == [i[0] for i in sorted_dict_L2])

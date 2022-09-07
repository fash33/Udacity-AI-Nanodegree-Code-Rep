import numpy as np
import pandas as pd


df = pd.read_csv('ml-bugs.csv')
def calc_entropy(m,n):
    p1 = m/(m+n)
    p2 = n/(m+n)
    entropy = - (p1 * np.log2(p1)) - (p2 * np.log2(p2))
    return entropy
def calc_entropy2(m,n,o):
    p1 = m/(m+n+o)
    p2 = n/(m+n+o)
    p3 = o/(m+n+o)
    entropy = - (p1 * np.log2(p1)) - (p2 * np.log2(p2)) - (p3 * np.log2(p3))
    return entropy

species_ent = calc_entropy(sum(df["Species"]== 'Lobug'),sum(df["Species"] == 'Mobug'))
color_ent = calc_entropy2(sum(df["Color"]== 'Brown'),sum(df["Color"] == 'Blue'),sum(df["Color"] == 'Green'))
len_ent = calc_entropy(sum(df["Length (mm)"] < 17),sum(df["Length (mm)"] > 17))

print(species_ent)
print(len_ent)
print("color inf_gain: %.3f \nlength inf_gain: %.3f" %(species_ent - (color_ent/2), species_ent - (len_ent)))


def two_group_ent(first, tot):
    return -(first/tot*np.log2(first/tot) +
             (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)
g17_ent = 15/24 * two_group_ent(11,15) + 9/24 * two_group_ent(6,9)

answer = tot_ent -g17_ent
print(g17_ent)
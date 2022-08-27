# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

def sublists(list1, list2):
    subs = []
    for i in range(len(list1)-1):
        for j in range(len(list2)-1):
            if list1[i]==list2[j] and list1[i+1]==list2[j+1]:
                m = i+2
                n = j+2
                while m<len(list1) and n<len(list2) and list1[m]==list2[n]:
                    m += 1
                    n += 1
                subs.append(list1[i:m])
    return subs

def max_sublists(list1, list2):
    subls = sublists(list1, list2)
    if len(subls)==0:
        return []
    else:
        max_len = max(len(subl) for subl in subls)
        return [subl for subl in subls if len(subl)==max_len]
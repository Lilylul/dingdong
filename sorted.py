def merge_sort(arr):
  if len(arr)<2:
    return arr
  
  mid = len(arr)//2
  low_arr = merge_sort(arr[:mid])
  high_arr = merge_sort(arr[mid:])

  merged_arr = []
  l = h = 0

  for i in range(len(low_arr)+len(high_arr)):
    if not low_arr:
      merged_arr.extend(high_arr)
      break
    if not high_arr:
      merged_arr.extend(low_arr)
      break
    if low_arr[0] <high_arr[0]:
      merged_arr.append(low_arr.pop(0))
    else:
      merged_arr.append(high_arr.pop(0))
  return merged_arr

rr= [1,2,3,4,13,9,6,7,8,15]
merge_sort(rr)



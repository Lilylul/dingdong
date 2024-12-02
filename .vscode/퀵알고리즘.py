# 리스트 arr 을 quick sort하는 함수
def quick_sort(arr):
    import random

    if len(arr) < 2: # 재귀적으로 나뉘어질때 리스트의 길이가 2보다 작으면 재귀적 호출을 중단하도록 하는 코드
        return arr
    
    pivot = random.randint(0,len(arr)-1) # 랜덤으로 피봇을 선정

    p_value = arr[pivot]

    temp_arr = [] # 정렬 완료된 요소들을 모두 저장할 리스트
    temp_left = [] # pivot 보다 작은 요소들을 저장할 리스트
    temp_right = [] # pivot 보다 큰 요소들을 저장할 리스트

    for i in range(pivot): # 왼쪽 리스트 정리
        if arr[i] > p_value:
            temp_right.append(arr[i]) # 피벗보다 큰 요소들을 temp_right에 저장
        else:
            temp_left.append(arr[i]) # 피벗보다 작거나 같은 요소들을 temp_left에 저장

    for i in range(pivot+1,len(arr)): # 오른쪽 리스트 정리
        if arr[i] > p_value:
            temp_right.append(arr[i]) # 피벗보다 작은 요소들을 temp_right에 저장
        else:
            temp_left.append(arr[i]) # 피벗보다 작은 요소들을 temp_right에 저장
        
    temp_left = quick_sort(temp_left) # 재귀적으로 피벗보다 작은 리스트만 정렬
    temp_right = quick_sort(temp_right) # 재귀적으로 피벗보다 큰 리스트만 정렬

    temp_arr = temp_left + [arr[pivot]] + temp_right # 정렬된 왼쪽리스트와 피벗과 오른쪽 리스트를 결합

    return temp_arr

arr = [3,5,1,4,9,8,7,10,1,3]

print(quick_sort(arr))
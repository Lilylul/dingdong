def merge_sort(arr):
    if len(arr)<2: #arr의 길이가 2보다 작으면
        return arr #arr를 반환하라
    mid = len(arr)//2 #mid는 arr 길이를 2로 나눈 값으로 할당한다.
    low_arr = merge_sort(arr[:mid]) #low는 arr의 0부터 mid 직전 값까지로 할당한다.
    high_arr = merge_sort(arr[mid:]) #high는 arr의 mid부터 끝값까지로 할당한다.

    merged_arr = [] #merged_arr를 리스트로 선언한다.
    l = h = 0 #l은 h이고 0이다.
    while l<len(low_arr) and h < len(high_arr): #low_arr의 길이는 l보다 크고, h는 high_arr 길이보다 작다.
        if low_arr[l] < high_arr[h]:
            merged_arr.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            h += 1
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    return merged_arr

n=[5,5,2,3,4,1]

print(merge_sort(n))


arr_1 = [3,3,5,1,3,4]

def solution(array):
    count = [0] * (max(array)+1) #index 번호는 0부터 시작임으로 +1을 지정 (배열의 값을 세기 위해)

    for i in array:
        count[i] += 1 #array의 값에 해당하는 인덱스에 1을 더하여 횟수를 기록
    m = 0 #맥스 변수 지정
    for c in count:  
        if c == max(count): #max(count)와 같은 값이 있으면 m에 추가
            m +=1

    if m >1 :
        answer = -1
    elif m == 1:
        answer = count.index(max(count)) 

    return answer    

print(solution(arr_1))



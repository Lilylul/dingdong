array = [10,5,3,12,7,6,5,3,2,4]
def quick_sort(array,start,end):
    if start >= end:
        return
#비교를 진행한다.
# 가장 왼쪽 배열의 인덱스를 저장한 left  변수,오른쪽 배열의 인덱스를 저장한 right 변수 생성
    pivot = start
    left = start+1
    right = end
    
    while left<=right: # left<right가 만족할때까지 반복
    # right부터 비교 한다. 비교는 right가 left보다 클때만 반복하며, 비교한 배열값이 pivot point 보다 크면  right를 하나 감소시키고 반복
    # # right 비교 시작
        while right > left and arr[pivot] <= arr[right]:
            right -= 1
    #left 비교 시작
        while right>left and arr[pivot] >= arr[left]:
            left += 1
        if left>right: #엇갈렸다면 작은 right -=1 데이터와 피벗을 교체
            array[right], array[pivot] = array[pivot], array[right]
        else: #엇갈리지 않았다면 작은 데이터와 큰데이터를 교체
        array[left],array[right] = array[right],array[left]

    #분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행    
    quick_sort(array, start, right-1)
    quick_sort(array, right+1, end)


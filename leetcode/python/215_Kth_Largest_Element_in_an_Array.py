# https://leetcode.com/problems/kth-largest-element-in-an-array/

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        min_heap = MinHeap(k)
        for num in nums:
            if num > min_heap.get_first_value():
                min_heap.replace(num)
            # min_heap.print_value()
        return min_heap.get_first_value()
        
class MinHeap(object):
    def __init__(self, k):
        """
        :type k: int
        :rtype: List[int]
        """
        self.min_heap = [-sys.maxint] * k
    
    def get_first_value(self):
        """
        rtype: int
        """
        return self.min_heap[0]
        
    def replace(self, n):
        """
        :type n: int
        """
        self.min_heap[0] = n
        self.reorganize(0)    
    
    def reorganize(self, n):
        """
        :type n: int
        """
        if n < len(self.min_heap):
            current_value = self.min_heap[n]
        else:
            return True
        left_child_idx = 2 * n + 1
        right_child_idx = 2 * n + 2
        if left_child_idx < len(self.min_heap):
            left_child_value = self.min_heap[left_child_idx]
        else:
            left_child_value = sys.maxint
        if right_child_idx < len(self.min_heap):
            right_child_value = self.min_heap[right_child_idx]
        else:
            right_child_value = sys.maxint
        if left_child_value < right_child_value:
            pos = left_child_idx
            smaller_child_value = left_child_value
        else:
            pos = right_child_idx
            smaller_child_value = right_child_value
        if current_value > smaller_child_value:
            self.min_heap[n] = smaller_child_value
            self.min_heap[pos] = current_value
            self.reorganize(pos)
        else:
            return True
        return True
    
    def print_value(self):
        """
        """
        print(self.min_heap)

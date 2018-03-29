// https://leetcode.com/problems/kth-largest-element-in-an-array/

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        MinHeap min_heap (k);
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] > min_heap.getFirstValue()) {
                min_heap.replace(nums[i]);
            }
            // min_heap.print();
        }
        return min_heap.getFirstValue();
    }
    
    class MinHeap {
    public:
        MinHeap(int k) {
            this->min_heap.resize(k, INT_MIN);
        }
        
        // ~MinHeap(); // adding this gives a compiling error!!!
        
        int getFirstValue() {
            return this->min_heap[0];
        }
        
        bool replace(int num) {
            this->min_heap[0] = num;
            this->reorganize(0);
            return true;
        }
        
        bool reorganize(int idx) {
            if (idx >= this->min_heap.size()) {
                return true;
            }
            int current_value = this->min_heap[idx];
            int left_child_idx = idx * 2 + 1;
            int right_child_idx = idx * 2 + 2;
            int left_child_value = INT_MAX;
            int right_child_value = INT_MAX;
            if (left_child_idx < this->min_heap.size()) {
                left_child_value = this->min_heap[left_child_idx];
            }
            if (right_child_idx < this->min_heap.size()) {
                right_child_value = this->min_heap[right_child_idx];
            }
            int min_child_idx = -1;
            int min_child_value = INT_MAX;
            if (left_child_value < right_child_value) {
                min_child_value = left_child_value;
                min_child_idx = left_child_idx;
            } else {
                min_child_value = right_child_value;
                min_child_idx = right_child_idx;
            }
            if (current_value > min_child_value) {
                this->min_heap[idx] = min_child_value;
                this->min_heap[min_child_idx] = current_value;
                this->reorganize(min_child_idx);
            }
            return true;
        }
        
        void print() {
            for (int i = 0; i < this->min_heap.size(); ++i) {
                cout << this->min_heap[i] << " ";
            }
            cout << endl;
        }
    private:
        vector<int> min_heap;
    };
};

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <list>

using namespace std;

struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class ListNode{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution{
public:
    int maxDepth(TreeNode* root){
        if(root == NULL) return 0;
        int leftHeight = maxDepth(root -> left);
        int rightHeight = maxDepth(root -> right);
        int height = max(leftHeight, rightHeight) + 1;
        return height;
    }

    int getDepth(TreeNode *root){
        if(!root) return 0;
        return max(getDepth(root -> left), getDepth(root -> right)) + 1;
    }

    void printUtil(TreeNode *root, vector<vector<string>> &ans, int start, int end, int depth){
        if(!root) return;
        int mid = start + (end - start)/2;
        ans[depth][mid] = to_string(root -> val);
        printUtil(root -> left, ans, start, mid - 1, depth + 1);
        printUtil(root -> right, ans, mid + 1, end, depth + 1);
    }

    vector<vector<string>> printTree(TreeNode* root){
        int d = getDepth(root);
        vector<vector<string>> ans(d, vector<string>(pow(2, d) - 1, ""));
        printUtil(root, ans, 0, pow(2, d) - 1, 0);
        return ans;
    }

    bool isValidBST(TreeNode* root, long min = LONG_MIN, long max = LONG_MAX){
        if(!root) return true;
        if(root -> val <= min || root -> val >= max) return false;
        return isValidBST(root -> left, min, root -> val) && isValidBST(root -> right, root -> val, max);
    }


//End of tree questions
    void moveZeroes(vector<int>& nums){
        int count = 0;
        for(int i = 0; i < nums.size(); i++){
            if(nums.at(i) != 0){
                nums.at(count++) = nums.at(i);
            }
            while(count > nums.size()){
                nums.at(count++) = 0;
            }
        }
    }

    int uniquePaths(int m, int n){
        int dp[m][n];
        for(int i = 0; i < m; i++){
            dp[i][0] = 1;
        }
        for(int i = 0; i < n; i++){
            dp[0][i] = 1;
        }

        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    void insertionSort(int arr[], int n){
        int i, key, j;
        for(i = 1; i < n; i++){
            key = arr[i];
            j = i - 1;

            while(j >= 0 && arr[j] > key){
                arr[j + 1] = arr[j];
                j--;
                arr[j + 1] = key;
            }
        }
    }

    int numTrees(int n){
        int BST[n + 2];
        BST[0] = 1;
        BST[1] = 1;

        for(int i = 2; i <= n; i++){
            BST[i] = 0;
            for(int j = 0; j < i; j++){
                BST[i] += BST[j] * BST[i - j - 1];
            }
        }return BST[n];
    }

    int searchInsert(vector<int>& nums, int target){
        if(target > nums.at(nums.size() - 1)) return nums.size();
        int l = 0;
        int r = nums.size() - 1;

        while(l < r){
            int m = l + (r - l)/2;
            if(target > nums.at(m)){
                l = m + 1;
            }else{
                r = m;
            }
        }
        return l;
    }
    //sum root to leaf leetcode challenge
    int helper(TreeNode* root, int sum){
        if(root == NULL) return 0;
        sum = sum*10 + root -> val;
        if(root -> left == NULL && root -> right == NULL){
            return sum;
        }
        return helper(root -> left, sum) + helper(root -> right, sum);
    }
    int sumNumbers(TreeNode* root) {
        return helper(root, 0);
    }
};

class graph {
    int V; //number of vertices
    list<int> *adjList; //adjacency list
    void DFS_helper(int v, bool visited[]);
public:
    //constructor
    graph(int V){
        this -> V = V; //this pointer referring to the current object
        adjList = new list<int>[V];
    }

    //adding an edge to the graph
    void addEdge(int v, int w){
        adjList[v].push_back(w);
    }
    void DFS(); //calling the traversal function
    void BFS(int key);
};

void graph::DFS_helper(int v, bool visited[]){
    //current node v is visited
    visited[v] = true;
    cout << v << " ";
    //recursively process all the adjacent vertices of the node
    list<int>::iterator i;
    for(i = adjList[v].begin(); i != adjList[v].end(); i++){
        if(! visited[*i])
            DFS_helper(*i, visited);
    }
}

void graph::DFS(){
    //initially none are visited
    bool *visited = new bool[V];
    for(int i = 0; i < V; i++){
        visited[i] = false;
    }
    // explore the vertices one by one by recursively calling DFS_helper
    for(int i = 0; i < V; i++){
        if(!visited[i]) DFS_helper(i, visited);
    }
}

void graph::BFS(int s){
    bool *visited = new bool[V];
    for(int i = 0; i < V; i++){
        visited[i] = false;
    }
    list<int> queue;
    visited[s] = true;
    queue.push_back(s);
    list<int>::iterator i;

    while(!queue.empty()){
        s = queue.front();
        cout << s << " ";
        queue.pop_front();

        for(i = adjList[s].begin(); i != adjList[s].end(); i++){
            if(!visited[*i]){
                visited[*i] = true;
                queue.push_back(*i);
            }
        }
    }
}


int main(){
    graph gdfs(5);
    gdfs.addEdge(0, 1);
    gdfs.addEdge(0, 2);
    gdfs.addEdge(0, 3);
    gdfs.addEdge(1, 2);
    gdfs.addEdge(2, 4);
    gdfs.addEdge(3, 3);
    gdfs.addEdge(4, 4);
    cout << "DFS traversal of the following graph \n";
    gdfs.DFS();
    cout << "\n";
    int vertex = 0;
    cout << "BFS traversal starting from vertex: " << vertex << "\n";
    gdfs.BFS(vertex);

    Solution sol;
    vector<int> zeroes = {1,0,0,2,3,4};
    sol.moveZeroes(zeroes);

    int arr[] = {4,3,6,1,7,2,9};
    int n = 7;
    sol.insertionSort(arr, n);
    for (const auto& e : arr) {
        cout << e << endl;
    }
    return 0;
}
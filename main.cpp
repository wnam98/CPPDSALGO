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

    Solution sol;
    vector<int> zeroes = {1,0,0,2,3,4};
    sol.moveZeroes(zeroes);
    return 0;
}
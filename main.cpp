#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <unordered_map>
#include <cmath>
#include <list>

using namespace std;

class Node {
public:
    Node(): last_char(false), child(vector<Node*>(26, nullptr)) {}
    bool last_char;
    vector<Node*> child;
};

class Trie {
public:
    Trie(int n, int m): vis(vector<vector<bool>>(n, vector<bool>(m, false))), root(new Node()) {
    }
    ~Trie() {
        del_tree(root);
    }
    void insert(const string &s) {
        Node* cur_node = root;
        for (int i = 0; i < s.size(); ++i) {
            if (cur_node->child[s[i] - 'a'] == nullptr) {
                cur_node->child[s[i] - 'a'] = new Node();
            }
            cur_node = cur_node->child[s[i] - 'a'];
            if (i == s.size() - 1) cur_node->last_char = true;
        }
    }
    void grid_search(const vector<vector<char>>& board, int i, int j, vector<string>& ret) {
        if (root->child[board[i][j] - 'a'] == nullptr) return ;
        Node* cur_node = root->child[board[i][j] - 'a'];
        string cur_str;
        dfs(board, i, j, ret, cur_node, cur_str);
    }
private:
    void dfs(const vector<vector<char>>& board, int i, int j, vector<string>& ret, Node* cur_node,
             string& cur_str) {
        vis[i][j] = true;
        cur_str.push_back(board[i][j]);
        if (cur_node->last_char) {
            ret.push_back(cur_str);
            cur_node->last_char = false;
        }
        for (int k = 0; k < 4; ++k) {
            int new_i = i + di[k], new_j = j + dj[k];
            if (new_i >= 0 && new_i < board.size() && new_j >= 0 && new_j < board[0].size()
                && !vis[new_i][new_j]
                && cur_node->child[board[new_i][new_j] - 'a'] != nullptr) {
                dfs(board, new_i, new_j, ret, cur_node->child[board[new_i][new_j] - 'a'],
                    cur_str);
            }
        }
        vis[i][j] = false;
        cur_str.pop_back();
    }
    void del_tree(Node* node) {
        for (int i = 0; i < 26; ++i) {
            if (node->child[i] != nullptr) del_tree(node->child[i]);
        }
        delete node;
    }
    Node* root;
    vector<vector<bool>> vis;
    int di[4] = {0, 1, 0, -1}, dj[4] = {1, 0, -1, 0};
};


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

    //coin change dp problem
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount + 1);
        dp.at(0) = 1;

        for(int coin: coins){
            for(int i = coin; i <= amount; i++){
                dp.at(i) += dp.at(i - coin);
            }
        }return dp.at(amount);
    }

    //reconstruct itinerary graph problem
    void dfs(string src,unordered_map<string,vector<string>>& m,vector<string>& ans){
        while(!m[src].empty()){
            string s=m[src].back();
            m[src].pop_back();
            dfs(s,m,ans);
        }
        ans.push_back(src);
    }

    vector<string> findItinerary(vector<vector<string>>& tickets) {
        unordered_map<string,vector<string>> m;
        // Create Graph
        for(auto i:tickets)
            m[i[0]].push_back(i[1]);
        // Sorting in descending order since we will be popping elements from the end
        for(auto &i:m)
            sort(i.second.begin(),i.second.end(),greater<string>());

        vector<string> ans;
        dfs("JFK",m,ans);
        reverse(ans.begin(),ans.end());
        return ans;
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        vector<string> ret;
        if (board.size() == 0 || board[0].size() == 0 || words.size() == 0) return ret;
        int n = board.size(), m = board[0].size();
        Trie trie(n, m);
        for (auto& word: words) {
            trie.insert(word);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                trie.grid_search(board, i, j, ret);
            }
        }
        return ret;
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